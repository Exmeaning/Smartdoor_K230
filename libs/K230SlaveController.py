"""
K230 从机控制器
单线程状态机模式，避免多线程问题
"""

import time
import gc
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart

class K230SlaveController:
    """K230从机主控制器 - 单线程状态机"""
    
    def __init__(self, baudrate=115200):
        # 通信模块
        self.uart = K230Uart(baudrate=baudrate)
        self.protocol = K230Protocol()
        
        # 状态管理
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        self.running = False
        self.stop_flag = False
        
        # 功能模块注册表
        self.func_handlers = {}           # 循环处理函数
        self.func_init_handlers = {}      # 初始化函数
        self.func_deinit_handlers = {}    # 清理函数
        self.func_once_handlers = {}      # 一次性执行函数（如注册）
        
        # 命令处理器
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
        
        # 当前功能对象
        self.current_func_obj = None
        
        # 命令队列（用于在功能运行时处理命令）
        self.pending_stop = False
        
        print("[Controller] Initialized (Single-thread mode)")
    
    # ========== 功能注册 ==========
    def register_function(self, func_id, handler, init_handler=None, deinit_handler=None):
        """注册循环执行的功能模块"""
        self.func_handlers[func_id] = handler
        if init_handler:
            self.func_init_handlers[func_id] = init_handler
        if deinit_handler:
            self.func_deinit_handlers[func_id] = deinit_handler
        print("[Controller] Registered function:", func_id)
    
    def register_once_function(self, func_id, handler):
        """注册一次性执行的功能（如人脸注册）"""
        self.func_once_handlers[func_id] = handler
        print("[Controller] Registered once-function:", func_id)
    
    # ========== 数据发送 ==========
    def send_response(self, rsp_type, msg="", data=None):
        """发送响应"""
        rsp = self.protocol.build_response(rsp_type, msg, data)
        self.uart.send(rsp)
    
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
            return False, "Unknown function:%d" % func_id
        
        if self.state == K230Protocol.STATE_RUNNING:
            # 标记停止，让当前功能退出
            self.pending_stop = True
            return False, "BUSY,stop current first"
        
        # 设置要启动的功能
        self.current_func_id = func_id
        self.stop_flag = False
        self.pending_stop = False
        
        return True, "Starting:%d" % func_id
    
    def _handle_stop(self, params):
        """处理STOP命令"""
        if self.state != K230Protocol.STATE_RUNNING:
            return True, "Already stopped"
        
        self.pending_stop = True
        self.stop_flag = True
        return True, "Stopping"
    
    def _handle_status(self, params):
        """处理STATUS命令"""
        status_str = "%d,%d" % (self.state, self.current_func_id)
        return True, status_str
    
    def _handle_ping(self, params):
        """处理PING命令"""
        self.send_response(K230Protocol.RSP_PONG, "K230")
        return None, None
    
    def _handle_reset(self, params):
        """处理RESET命令"""
        self.pending_stop = True
        self.stop_flag = True
        gc.collect()
        return True, "Reset"
    
    def _handle_register(self, params):
        """处理REG命令"""
        if len(params) < 2:
            return False, "Usage:REG,user_id,photo_path"
        
        user_id = params[0]
        photo_path = params[1]
        
        if K230Protocol.ID_FACE_REGISTER in self.func_once_handlers:
            if self.state == K230Protocol.STATE_RUNNING:
                return False, "BUSY,stop AI first"
            
            self.state = K230Protocol.STATE_BUSY
            try:
                handler = self.func_once_handlers[K230Protocol.ID_FACE_REGISTER]
                success, msg = handler(self, user_id, photo_path)
                return success, msg
            except Exception as e:
                return False, str(e)
            finally:
                self.state = K230Protocol.STATE_IDLE
        else:
            return False, "Register not available"
    
    def _handle_list(self, params):
        """处理LIST命令"""
        import os
        try:
            db_dir = self.config.get('database_dir', '/data/face_database/')
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
        """处理DELETE命令"""
        if len(params) < 1:
            return False, "Usage:DELETE,user_id"
        
        import os
        user_id = params[0]
        db_dir = self.config.get('database_dir', '/data/face_database/')
        
        try:
            bin_path = db_dir + user_id + ".bin"
            os.remove(bin_path)
            return True, "Deleted:%s" % user_id
        except:
            return False, "Not found"
    
    def _handle_set(self, params):
        """处理SET命令"""
        if len(params) < 2:
            return False, "Usage:SET,key,value"
        
        key = params[0]
        value = params[1]
        
        try:
            if '.' in value:
                value = float(value)
            else:
                value = int(value)
        except:
            pass
        
        self.config[key] = value
        return True, "%s=%s" % (key, str(value))
    
    def _handle_get(self, params):
        """处理GET命令"""
        if len(params) < 1:
            return False, "Usage:GET,key"
        
        key = params[0]
        if key in self.config:
            return True, "%s=%s" % (key, str(self.config[key]))
        else:
            return False, "Unknown key"
    
    # ========== 命令处理循环 ==========
    def check_and_process_command(self):
        """
        检查并处理命令（非阻塞）
        返回: True 如果需要停止当前功能
        """
        cmd_str = self.uart.receive_command()
        if cmd_str is None:
            return self.pending_stop
        
        # 解析命令
        cmd, params = self.protocol.parse_command(cmd_str)
        if cmd is None:
            return self.pending_stop
        
        print("[Controller] CMD:", cmd, params)
        
        # 在运行状态下，只处理 STOP/STATUS/PING
        if self.state == K230Protocol.STATE_RUNNING:
            if cmd == K230Protocol.CMD_STOP:
                self.pending_stop = True
                self.stop_flag = True
                self.send_response(K230Protocol.RSP_OK, "Stopping")
            elif cmd == K230Protocol.CMD_STATUS:
                result = self._handle_status(params)
                self.send_response(K230Protocol.RSP_OK, result[1])
            elif cmd == K230Protocol.CMD_PING:
                self._handle_ping(params)
            else:
                self.send_response(K230Protocol.RSP_BUSY, "Running func:%d" % self.current_func_id)
            return self.pending_stop
        
        # 空闲状态，处理所有命令
        if cmd in self.cmd_handlers:
            result = self.cmd_handlers[cmd](params)
            if result and result[0] is not None:
                success, msg = result
                if success:
                    self.send_response(K230Protocol.RSP_OK, msg)
                else:
                    self.send_response(K230Protocol.RSP_ERROR, msg)
        else:
            self.send_response(K230Protocol.RSP_ERROR, "Unknown:%s" % cmd)
        
        return self.pending_stop
    
    # ========== 功能执行 ==========
    def _init_function(self, func_id):
        """初始化功能"""
        if func_id in self.func_init_handlers:
            print("[Controller] Init function:", func_id)
            self.current_func_obj = self.func_init_handlers[func_id](self)
            return True
        return False
    
    def _run_function_once(self, func_id):
        """执行一帧功能"""
        if func_id in self.func_handlers:
            handler = self.func_handlers[func_id]
            # 传入 stop_check 回调
            handler(self, lambda: self.stop_flag)
    
    def _deinit_function(self, func_id):
        """清理功能"""
        if func_id in self.func_deinit_handlers and self.current_func_obj:
            print("[Controller] Deinit function:", func_id)
            try:
                self.func_deinit_handlers[func_id](self.current_func_obj)
            except Exception as e:
                print("[Controller] Deinit error:", e)
        self.current_func_obj = None
        gc.collect()
    
    # ========== 主循环 ==========
    def run(self):
        """主运行循环 - 单线程状态机"""
        self.running = True
        self.uart.clear_buffer()
        
        print("[Controller] Running...")
        self.send_response(K230Protocol.RSP_READY, "K230")
        
        while self.running:
            try:
                # 空闲状态：等待命令
                if self.state == K230Protocol.STATE_IDLE:
                    self.check_and_process_command()
                    
                    # 检查是否有功能要启动
                    if self.current_func_id != 0 and not self.stop_flag:
                        func_id = self.current_func_id
                        
                        # 初始化功能
                        try:
                            self._init_function(func_id)
                            self.state = K230Protocol.STATE_RUNNING
                            self.send_response(K230Protocol.RSP_OK, "Started:%d" % func_id)
                            print("[Controller] Started function:", func_id)
                        except Exception as e:
                            print("[Controller] Init error:", e)
                            self.send_response(K230Protocol.RSP_ERROR, str(e))
                            self.current_func_id = 0
                    
                    time.sleep_ms(10)
                
                # 运行状态：执行功能 + 检查命令
                elif self.state == K230Protocol.STATE_RUNNING:
                    func_id = self.current_func_id
                    
                    # 执行一帧
                    try:
                        self._run_function_once(func_id)
                    except Exception as e:
                        print("[Controller] Run error:", e)
                        self.pending_stop = True
                    
                    # 检查命令（非阻塞）
                    self.check_and_process_command()
                    
                    # 检查是否需要停止
                    if self.pending_stop or self.stop_flag:
                        print("[Controller] Stopping function:", func_id)
                        self._deinit_function(func_id)
                        self.state = K230Protocol.STATE_IDLE
                        self.current_func_id = 0
                        self.pending_stop = False
                        self.stop_flag = False
                        self.send_response(K230Protocol.RSP_OK, "Stopped")
                
                else:
                    # 其他状态，检查命令
                    self.check_and_process_command()
                    time.sleep_ms(10)
                
            except KeyboardInterrupt:
                print("[Controller] Keyboard interrupt")
                break
            except Exception as e:
                print("[Controller] Error:", e)
                time.sleep_ms(100)
        
        # 清理
        if self.state == K230Protocol.STATE_RUNNING:
            self._deinit_function(self.current_func_id)
        
        print("[Controller] Stopped")
    
    def stop(self):
        """停止控制器"""
        self.running = False
        self.pending_stop = True
        self.stop_flag = True
    
    def deinit(self):
        """释放资源"""
        self.stop()
        self.uart.deinit()
        print("[Controller] Deinitialized")