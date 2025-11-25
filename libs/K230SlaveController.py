"""
K230 从机控制器 - 简化版
摄像头常驻，只切换AI模型
"""

import time
import gc
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart
from libs.CameraManager import CameraManager
from libs.AIProcessor import AIProcessor

class K230SlaveController:
    """K230从机控制器"""
    
    def __init__(self, baudrate=115200):
        self.uart = K230Uart(baudrate=baudrate)
        self.protocol = K230Protocol()
        
        # 状态
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        self.running = False
        self.stop_flag = False
        
        # 摄像头（常驻）
        self.camera = CameraManager.get_instance()
        
        # AI处理器
        self.ai = None
        
        # 命令处理器
        self.cmd_handlers = {
            K230Protocol.CMD_START: self._handle_start,
            K230Protocol.CMD_STOP: self._handle_stop,
            K230Protocol.CMD_STATUS: self._handle_status,
            K230Protocol.CMD_PING: self._handle_ping,
            K230Protocol.CMD_RESET: self._handle_reset,
            K230Protocol.CMD_SET: self._handle_set,
            K230Protocol.CMD_GET: self._handle_get,
            K230Protocol.CMD_LIST: self._handle_list,
            K230Protocol.CMD_DELETE: self._handle_delete,
            K230Protocol.CMD_REGCAM: self._handle_register_camera,
            K230Protocol.CMD_RELOAD: self._handle_reload,
        }
        
        # 功能ID映射到模型类型
        self.func_to_model = {
            K230Protocol.ID_FACE_DETECT: AIProcessor.MODEL_FACE_DETECT,
            K230Protocol.ID_FACE_RECOGNITION: AIProcessor.MODEL_FACE_RECOGNIZE,
        }
        
        # 配置
        self.config = {
            'face_threshold': 0.65,
            'detect_threshold': 0.5,
            'nms_threshold': 0.2,
            'database_dir': '/data/face_database/',
            'idle_fps': 5,
            'active_fps': 30,
        }
        
        # 注册处理器
        self.camera_register_handler = None
        
        print("[Controller] Initialized")
    
    def register_camera_handler(self, handler):
        self.camera_register_handler = handler
    
    # ========== 数据发送 ==========
    def send_response(self, rsp_type, msg="", data=None):
        rsp = self.protocol.build_response(rsp_type, msg, data)
        self.uart.send(rsp)
    
    def send_face_detect(self, x, y, w, h):
        self.uart.send(self.protocol.get_face_detect_data(x, y, w, h))
    
    def send_face_recognition(self, x, y, w, h, name, score):
        self.uart.send(self.protocol.get_face_recognition_data(x, y, w, h, name, score))
    
    def send_register_result(self, success, user_id, msg=""):
        self.uart.send(self.protocol.get_register_result_data(success, user_id, msg))
    
    # ========== 命令处理 ==========
    def _handle_start(self, params):
        if len(params) < 1:
            return False, "Missing func_id"
        
        try:
            func_id = int(params[0])
        except:
            return False, "Invalid func_id"
        
        if func_id not in self.func_to_model:
            return False, "Unknown function:%d" % func_id
        
        if self.state == K230Protocol.STATE_RUNNING:
            if self.current_func_id == func_id:
                return True, "Already running"
            # 切换功能：先停止当前
            self._do_stop()
        
        # 加载模型
        model_type = self.func_to_model[func_id]
        success = self.ai.load(
            model_type,
            det_threshold=self.config['detect_threshold'],
            nms_threshold=self.config['nms_threshold'],
            face_threshold=self.config['face_threshold'],
            database_dir=self.config['database_dir']
        )
        
        if not success:
            return False, "Load model failed"
        
        # 提高帧率
        self.camera.set_fps(self.config['active_fps'])
        
        self.current_func_id = func_id
        self.state = K230Protocol.STATE_RUNNING
        self.stop_flag = False
        
        return True, "Started:%d" % func_id
    
    def _handle_stop(self, params):
        if self.state == K230Protocol.STATE_IDLE:
            return True, "Already stopped"
        
        self._do_stop()
        return True, "Stopped"
    
    def _do_stop(self):
        """执行停止"""
        self.stop_flag = True
        self.ai.unload()
        
        # 降低帧率
        self.camera.set_fps(self.config['idle_fps'])
        
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
    
    def _handle_status(self, params):
        return True, "%d,%d" % (self.state, self.current_func_id)
    
    def _handle_ping(self, params):
        self.send_response(K230Protocol.RSP_PONG, "K230")
        return None, None
    
    def _handle_reset(self, params):
        self._do_stop()
        gc.collect()
        return True, "Reset"
    
    def _handle_set(self, params):
        if len(params) < 2:
            return False, "Usage:SET,key,value"
        
        key, value = params[0], params[1]
        
        # 特殊处理帧率
        if key == 'fps':
            try:
                fps = int(value)
                self.camera.set_fps(fps)
                return True, "fps=%d" % fps
            except:
                return False, "Invalid fps"
        
        # 其他配置
        try:
            value = float(value) if '.' in value else int(value)
        except:
            pass
        
        self.config[key] = value
        return True, "%s=%s" % (key, str(value))
    
    def _handle_get(self, params):
        if len(params) < 1:
            return False, "Usage:GET,key"
        
        key = params[0]
        
        if key == 'fps':
            return True, "fps=%d" % self.camera.get_fps()
        
        if key in self.config:
            return True, "%s=%s" % (key, str(self.config[key]))
        
        return False, "Unknown key"
    
    def _handle_list(self, params):
        import os
        try:
            db_dir = self.config['database_dir']
            users = [f[:-4] for f in os.listdir(db_dir) if f.endswith('.bin')]
            return True, ",".join(users) if users else "empty"
        except:
            return True, "empty"
    
    def _handle_delete(self, params):
        if len(params) < 1:
            return False, "Usage:DELETE,user_id"
        
        import os
        try:
            os.remove(self.config['database_dir'] + params[0] + ".bin")
            return True, "Deleted"
        except:
            return False, "Not found"
    
    def _handle_register_camera(self, params):
        if len(params) < 1:
            return False, "Usage:REGCAM,user_id[,timeout]"
        
        if self.state == K230Protocol.STATE_RUNNING:
            return False, "BUSY,stop first"
        
        if self.camera_register_handler is None:
            return False, "Not available"
        
        user_id = params[0]
        timeout = int(params[1]) if len(params) > 1 else 10
        
        self.state = K230Protocol.STATE_REGISTERING
        self.send_response(K230Protocol.RSP_PROGRESS, "Registering...")
        
        # 提高帧率用于注册
        self.camera.set_fps(30)
        
        try:
            success, msg = self.camera_register_handler(self, user_id, timeout)
            self.send_register_result(success, user_id, msg)
            return success, msg
        finally:
            self.camera.set_fps(self.config['idle_fps'])
            self.state = K230Protocol.STATE_IDLE
    
    def _handle_reload(self, params):
        if self.ai and self.ai.is_loaded():
            self.ai.reload_database()
        return True, "Reloaded"
    
    # ========== 命令处理循环 ==========
    def _process_command(self):
        cmd_str = self.uart.receive_command()
        if cmd_str is None:
            return
        
        cmd, params = self.protocol.parse_command(cmd_str)
        if cmd is None:
            return
        
        print("[CMD]", cmd, params)
        
        if cmd in self.cmd_handlers:
            result = self.cmd_handlers[cmd](params)
            if result and result[0] is not None:
                success, msg = result
                rsp = K230Protocol.RSP_OK if success else K230Protocol.RSP_ERROR
                self.send_response(rsp, msg)
        else:
            self.send_response(K230Protocol.RSP_ERROR, "Unknown:%s" % cmd)
    
    # ========== AI处理循环 ==========
    def _process_ai(self):
        """处理一帧AI"""
        if self.state != K230Protocol.STATE_RUNNING:
            return
        
        frame = self.camera.get_frame()
        if frame is None:
            return
        
        results = self.ai.process(frame)
        
        # 清空OSD
        self.camera.clear_osd()
        osd = self.camera.get_osd()
        
        rgb_size = self.camera.get_rgb_size()
        disp_size = self.camera.get_display_size()
        
        # 绘制结果并发送
        for r in results:
            if self.current_func_id == K230Protocol.ID_FACE_DETECT:
                x, y, w, h = r
                # 坐标转换
                x_d = x * disp_size[0] // rgb_size[0]
                y_d = y * disp_size[1] // rgb_size[1]
                w_d = w * disp_size[0] // rgb_size[0]
                h_d = h * disp_size[1] // rgb_size[1]
                
                osd.draw_rectangle(x_d, y_d, w_d, h_d, color=(255, 255, 0, 255), thickness=2)
                self.send_face_detect(x, y, w, h)
                
            elif self.current_func_id == K230Protocol.ID_FACE_RECOGNITION:
                x, y, w, h, name, score = r
                x_d = x * disp_size[0] // rgb_size[0]
                y_d = y * disp_size[1] // rgb_size[1]
                w_d = w * disp_size[0] // rgb_size[0]
                h_d = h * disp_size[1] // rgb_size[1]
                
                color = (255, 0, 255, 0) if name != "unknown" else (255, 255, 0, 0)
                osd.draw_rectangle(x_d, y_d, w_d, h_d, color=color, thickness=2)
                
                text = "%s:%.2f" % (name, score) if name != "unknown" else "unknown"
                osd.draw_string_advanced(x_d, y_d - 20, 18, text, color=(255, 255, 255, 0))
                
                self.send_face_recognition(x, y, w, h, name, score)
        
        self.camera.show()
    
    # ========== 主循环 ==========
    def run(self):
        """主循环"""
        print("[Controller] Starting...")
        
        # 启动摄像头
        if not self.camera.start(fps=self.config['idle_fps']):
            print("[Controller] Camera start failed!")
            return
        
        # 创建AI处理器
        self.ai = AIProcessor(
            rgb_size=self.camera.get_rgb_size(),
            display_size=self.camera.get_display_size()
        )
        
        self.running = True
        self.send_response(K230Protocol.RSP_READY, "K230")
        print("[Controller] Running...")
        
        frame_count = 0
        
        while self.running:
            try:
                # 处理命令
                self._process_command()
                
                # 处理AI（如果在运行）
                if self.state == K230Protocol.STATE_RUNNING:
                    self._process_ai()
                else:
                    # 空闲时也要刷新显示（否则画面卡住）
                    self.camera.get_frame()
                    self.camera.clear_osd()
                    self.camera.show()
                
                frame_count += 1
                if frame_count % 100 == 0:
                    gc.collect()
                
                # 短暂延时
                time.sleep_ms(1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("[Controller] Error:", e)
                time.sleep_ms(100)
        
        # 清理
        self.ai.unload()
        self.camera.stop()
        print("[Controller] Stopped")
    
    def stop(self):
        self.running = False
        self.stop_flag = True
    
    def deinit(self):
        self.stop()
        self.uart.deinit()