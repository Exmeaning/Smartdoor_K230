"""
K230 从机控制器
管理命令接收、功能调度和状态维护
"""

import time
import gc
import _thread
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart

class K230SlaveController:
    """K230从机主控制器"""
    
    def __init__(self, baudrate=115200):
        """
        初始化控制器
        
        参数:
            baudrate: 波特率
        """
        # 通信模块
        self.uart = K230Uart(baudrate=baudrate)
        self.protocol = K230Protocol()
        
        # 状态管理
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        self.running = False
        self.stop_flag = False
        
        # 功能模块注册表
        self.func_handlers = {}
        self.func_init_handlers = {}
        self.func_deinit_handlers = {}
        
        # 命令处理器注册表
        self.cmd_handlers = {
            K230Protocol.CMD_START: self._handle_start,
            K230Protocol.CMD_STOP: self._handle_stop,
            K230Protocol.CMD_STATUS: self._handle_status,
            K230Protocol.CMD_PING: self._handle_ping,
            K230Protocol.CMD_RESET: self._handle_reset,
            K230Protocol.CMD_REG: self._handle_register,
            K230Protocol.CMD_LIST: self._handle_list,
            K230Protocol.CMD_DELETE: self._handle_delete,
            K230Protocol.CMD_SET: self._handle_set,
            K230Protocol.CMD_GET: self._handle_get,
        }
        
        # 配置参数
        self.config = {
            'face_threshold': 0.65,
            'detect_threshold': 0.5,
            'nms_threshold': 0.2,
            'database_dir': '/data/face_database/',
        }
        
        # 线程锁
        self.lock = _thread.allocate_lock()
        
        # 当前功能对象
        self.current_func_obj = None
        
        print("[Controller] Initialized")
    
    # ========== 功能注册 ==========
    def register_function(self, func_id, handler, init_handler=None, deinit_handler=None):
        """
        注册功能模块
        
        参数:
            func_id: 功能ID
            handler: 功能处理函数 handler(controller, stop_flag_callback)
            init_handler: 初始化函数 init_handler(controller) -> obj
            deinit_handler: 清理函数 deinit_handler(obj)
        """
        self.func_handlers[func_id] = handler
        if init_handler:
            self.func_init_handlers[func_id] = init_handler
        if deinit_handler:
            self.func_deinit_handlers[func_id] = deinit_handler
        print("[Controller] Registered function:", func_id)
    
    def register_command(self, cmd_type, handler):
        """
        注册自定义命令处理器
        
        参数:
            cmd_type: 命令类型字符串
            handler: 处理函数 handler(params) -> (success, message)
        """
        self.cmd_handlers[cmd_type] = handler
    
    # ========== 数据发送 ==========
    def send_response(self, rsp_type, msg="", data=None):
        """发送响应"""
        rsp = self.protocol.build_response(rsp_type, msg, data)
        self.uart.send(rsp)
    
    def send_data(self, data_packet):
        """发送数据包"""
        self.uart.send(data_packet)
    
    def send_face_detect(self, x, y, w, h):
        """发送人脸检测结果"""
        packet = self.protocol.get_face_detect_data(x, y, w, h)
        self.uart.send(packet)
    
    def send_face_recognition(self, x, y, w, h, name, score):
        """发送人脸识别结果"""
        packet = self.protocol.get_face_recognition_data(x, y, w, h, name, score)
        self.uart.send(packet)
    
    def send_person_detect(self, x, y, w, h):
        """发送人体检测结果"""
        packet = self.protocol.get_person_detect_data(x, y, w, h)
        self.uart.send(packet)
    
    def send_object_detect(self, x, y, w, h, label):
        """发送物体检测结果"""
        packet = self.protocol.get_object_detect_data(x, y, w, h, label)
        self.uart.send(packet)
    
    # ========== 命令处理 ==========
    def _handle_start(self, params):
        """处理START命令"""
        if len(params) < 1:
            return False, "Missing func_id"
        
        try:
            func_id = int(params[0])
        except:
            return False, "Invalid func_id"
        
        if func_id not in self.func_handlers:
            return False, "Unknown function: %d" % func_id
        
        if self.state == K230Protocol.STATE_RUNNING:
            # 先停止当前功能
            self._stop_current_function()
        
        # 启动新功能
        self.current_func_id = func_id
        self.stop_flag = False
        self.state = K230Protocol.STATE_RUNNING
        
        # 在新线程中执行功能
        _thread.start_new_thread(self._run_function, (func_id,))
        
        return True, "Started:%d" % func_id
    
    def _handle_stop(self, params):
        """处理STOP命令"""
        if self.state != K230Protocol.STATE_RUNNING:
            return True, "Already stopped"
        
        self._stop_current_function()
        return True, "Stopped"
    
    def _handle_status(self, params):
        """处理STATUS命令"""
        status_str = "%d,%d" % (self.state, self.current_func_id)
        return True, status_str
    
    def _handle_ping(self, params):
        """处理PING命令"""
        self.send_response(K230Protocol.RSP_PONG, "K230")
        return None, None  # 已经发送响应，不需要额外响应
    
    def _handle_reset(self, params):
        """处理RESET命令"""
        self._stop_current_function()
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        gc.collect()
        return True, "Reset complete"
    
    def _handle_register(self, params):
        """处理REG命令 (人脸注册)"""
        if len(params) < 2:
            return False, "Usage: REG,<user_id>,<photo_path>"
        
        user_id = params[0]
        photo_path = params[1]
        
        # 调用注册功能（如果已注册）
        if K230Protocol.ID_FACE_REGISTER in self.func_handlers:
            # 标记为忙碌
            self.state = K230Protocol.STATE_BUSY
            
            try:
                handler = self.func_handlers[K230Protocol.ID_FACE_REGISTER]
                success, msg = handler(self, user_id, photo_path)
                self.state = K230Protocol.STATE_IDLE
                return success, msg
            except Exception as e:
                self.state = K230Protocol.STATE_IDLE
                return False, str(e)
        else:
            return False, "Register function not available"
    
    def _handle_list(self, params):
        """处理LIST命令 (列出注册人脸)"""
        import os
        try:
            db_dir = self.config.get('database_dir', '/data/face_database/')
            
            # 获取用户目录列表
            try:
                os.stat(db_dir)
                users = []
                for item in os.listdir(db_dir):
                    if item.endswith('.bin'):
                        users.append(item[:-4])
                
                return True, ",".join(users) if users else "empty"
            except:
                return True, "empty"
        except Exception as e:
            return False, str(e)
    
    def _handle_delete(self, params):
        """处理DELETE命令 (删除注册人脸)"""
        if len(params) < 1:
            return False, "Usage: DELETE,<user_id>"
        
        import os
        user_id = params[0]
        db_dir = self.config.get('database_dir', '/data/face_database/')
        
        try:
            bin_path = db_dir + user_id + ".bin"
            
            try:
                os.remove(bin_path)
                return True, "Deleted:%s" % user_id
            except:
                return False, "User not found"
                
        except Exception as e:
            return False, str(e)
    
    def _handle_set(self, params):
        """处理SET命令 (设置参数)"""
        if len(params) < 2:
            return False, "Usage: SET,<key>,<value>"
        
        key = params[0]
        value = params[1]
        
        # 尝试转换数值
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            pass  # 保持字符串
        
        self.config[key] = value
        return True, "%s=%s" % (key, str(value))
    
    def _handle_get(self, params):
        """处理GET命令 (获取参数)"""
        if len(params) < 1:
            return False, "Usage: GET,<key>"
        
        key = params[0]
        if key in self.config:
            return True, "%s=%s" % (key, str(self.config[key]))
        else:
            return False, "Unknown key"
    
    # ========== 功能执行 ==========
    def _run_function(self, func_id):
        """在线程中运行功能"""
        try:
            # 初始化
            if func_id in self.func_init_handlers:
                self.current_func_obj = self.func_init_handlers[func_id](self)
            
            # 执行
            handler = self.func_handlers[func_id]
            handler(self, lambda: self.stop_flag)
            
        except Exception as e:
            print("[Controller] Function error:", e)
            self.send_response(K230Protocol.RSP_ERROR, str(e))
        
        finally:
            # 清理
            self._cleanup_function(func_id)
            self.state = K230Protocol.STATE_IDLE
            self.current_func_id = 0
    
    def _stop_current_function(self):
        """停止当前功能"""
        self.stop_flag = True
        # 等待功能结束
        timeout = 50  # 5秒超时
        while self.state == K230Protocol.STATE_RUNNING and timeout > 0:
            time.sleep_ms(100)
            timeout -= 1
    
    def _cleanup_function(self, func_id):
        """清理功能"""
        if func_id in self.func_deinit_handlers and self.current_func_obj:
            try:
                self.func_deinit_handlers[func_id](self.current_func_obj)
            except Exception as e:
                print("[Controller] Cleanup error:", e)
        self.current_func_obj = None
        gc.collect()
    
    # ========== 主循环 ==========
    def process_commands(self):
        """处理接收到的命令（非阻塞，在主循环中调用）"""
        line = self.uart.receive_line()
        if line is None:
            return
        
        # 解析命令
        cmd, params = self.protocol.parse_command(line)
        if cmd is None:
            return
        
        # 查找处理器
        if cmd in self.cmd_handlers:
            result = self.cmd_handlers[cmd](params)
            
            # 发送响应（如果处理器返回了结果）
            if result and result[0] is not None:
                success, msg = result
                if success:
                    self.send_response(K230Protocol.RSP_OK, msg)
                else:
                    self.send_response(K230Protocol.RSP_ERROR, msg)
        else:
            self.send_response(K230Protocol.RSP_ERROR, "Unknown command: %s" % cmd)
    
    def run(self):
        """
        主运行循环
        持续处理命令
        """
        self.running = True
        print("[Controller] Running...")
        self.send_response(K230Protocol.RSP_READY, "K230")
        
        while self.running:
            try:
                self.process_commands()
                time.sleep_us(1)  # 让出CPU时间
            except Exception as e:
                print("[Controller] Error:", e)
                time.sleep_ms(100)
    
    def stop(self):
        """停止控制器"""
        self.running = False
        self._stop_current_function()
    
    def deinit(self):
        """释放资源"""
        self.stop()
        self.uart.deinit()
        print("[Controller] Deinitialized")