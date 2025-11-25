"""
K230 从机控制器
单线程状态机模式，支持人脸注册
修复：添加 MediaHelper 支持
"""

import time
import gc
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart

class K230SlaveController:
    """K230从机主控制器"""
    
    def __init__(self, baudrate=115200):
        self.uart = K230Uart(baudrate=baudrate)
        self.protocol = K230Protocol()
        
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        self.running = False
        self.stop_flag = False
        
        self.func_handlers = {}
        self.func_init_handlers = {}
        self.func_deinit_handlers = {}
        self.func_once_handlers = {}
        
        self.cmd_handlers = {
            K230Protocol.CMD_START: self._handle_start,
            K230Protocol.CMD_STOP: self._handle_stop,
            K230Protocol.CMD_STATUS: self._handle_status,
            K230Protocol.CMD_PING: self._handle_ping,
            K230Protocol.CMD_RESET: self._handle_reset,
            K230Protocol.CMD_REG: self._handle_register_photo,
            K230Protocol.CMD_REGCAM: self._handle_register_camera,
            K230Protocol.CMD_LIST: self._handle_list,
            K230Protocol.CMD_DELETE: self._handle_delete,
            K230Protocol.CMD_SET: self._handle_set,
            K230Protocol.CMD_GET: self._handle_get,
            K230Protocol.CMD_RELOAD: self._handle_reload,
            K230Protocol.CMD_FRESET: self._handle_force_reset,
        }
        
        self.config = {
            'face_threshold': 0.65,
            'detect_threshold': 0.5,
            'nms_threshold': 0.2,
            'database_dir': '/data/face_database/',
            'register_timeout': 10,
        }
        
        self.current_func_obj = None
        self.pending_stop = False
        
        # 注册处理器引用
        self.photo_register_handler = None
        self.camera_register_handler = None
        
        print("[Controller] Initialized (Single-thread mode)")
    
    # ========== 功能注册 ==========
    def register_function(self, func_id, handler, init_handler=None, deinit_handler=None):
        self.func_handlers[func_id] = handler
        if init_handler:
            self.func_init_handlers[func_id] = init_handler
        if deinit_handler:
            self.func_deinit_handlers[func_id] = deinit_handler
        print("[Controller] Registered function:", func_id)
    
    def register_once_function(self, func_id, handler):
        self.func_once_handlers[func_id] = handler
        print("[Controller] Registered once-function:", func_id)
    
    def register_photo_handler(self, handler):
        """注册照片注册处理器"""
        self.photo_register_handler = handler
        print("[Controller] Registered photo register handler")
    
    def register_camera_handler(self, handler):
        """注册摄像头注册处理器"""
        self.camera_register_handler = handler
        print("[Controller] Registered camera register handler")
    
    # ========== 数据发送 ==========
    def send_response(self, rsp_type, msg="", data=None):
        rsp = self.protocol.build_response(rsp_type, msg, data)
        self.uart.send(rsp)
    
    def send_face_detect(self, x, y, w, h):
        packet = self.protocol.get_face_detect_data(x, y, w, h)
        self.uart.send(packet)
    
    def send_face_recognition(self, x, y, w, h, name, score):
        packet = self.protocol.get_face_recognition_data(x, y, w, h, name, score)
        self.uart.send(packet)
    
    def send_person_detect(self, x, y, w, h):
        packet = self.protocol.get_person_detect_data(x, y, w, h)
        self.uart.send(packet)
    
    def send_object_detect(self, x, y, w, h, label):
        packet = self.protocol.get_object_detect_data(x, y, w, h, label)
        self.uart.send(packet)
    
    def send_register_result(self, success, user_id, msg=""):
        packet = self.protocol.get_register_result_data(success, user_id, msg)
        self.uart.send(packet)
    
    # ========== 命令处理 ==========
    def _handle_start(self, params):
        if len(params) < 1:
            return False, "Missing func_id"
        
        try:
            func_id = int(params[0])
        except:
            return False, "Invalid func_id"
        
        if func_id not in self.func_handlers:
            return False, "Unknown function:%d" % func_id
        
        if self.state == K230Protocol.STATE_RUNNING:
            self.pending_stop = True
            return False, "BUSY,stop first"
        
        if self.state == K230Protocol.STATE_REGISTERING:
            return False, "BUSY,registering"
        
        self.current_func_id = func_id
        self.stop_flag = False
        self.pending_stop = False
        
        return True, "Starting:%d" % func_id
    
    def _handle_stop(self, params):
        if self.state == K230Protocol.STATE_IDLE:
            return True, "Already stopped"
        
        if self.state == K230Protocol.STATE_REGISTERING:
            return False, "Cannot stop registration"
        
        self.pending_stop = True
        self.stop_flag = True
        return True, "Stopping"
    
    def _handle_status(self, params):
        status_str = "%d,%d" % (self.state, self.current_func_id)
        return True, status_str
    
    def _handle_ping(self, params):
        self.send_response(K230Protocol.RSP_PONG, "K230")
        return None, None
    
    def _handle_reset(self, params):
        self.pending_stop = True
        self.stop_flag = True
        gc.collect()
        return True, "Reset"
    
    def _handle_force_reset(self, params):
        """处理 FRESET 命令（强制重置媒体）"""
        try:
            from libs.MediaHelper import MediaHelper
            
            self.pending_stop = True
            self.stop_flag = True
            self.state = K230Protocol.STATE_IDLE
            self.current_func_id = 0
            self.current_func_obj = None
            
            MediaHelper.force_reset()
            
            return True, "Force reset done"
        except Exception as e:
            return False, str(e)
    
    def _handle_register_photo(self, params):
        """处理REG命令（从照片注册）"""
        if len(params) < 2:
            return False, "Usage:REG,user_id,photo_path"
        
        user_id = params[0]
        photo_path = params[1]
        
        if self.state == K230Protocol.STATE_RUNNING:
            return False, "BUSY,stop AI first"
        
        if self.state == K230Protocol.STATE_REGISTERING:
            return False, "BUSY,already registering"
        
        if self.photo_register_handler is None:
            return False, "Photo register not available"
        
        # 设置状态
        self.state = K230Protocol.STATE_REGISTERING
        self.send_response(K230Protocol.RSP_PROGRESS, "Registering from photo...")
        
        try:
            success, msg = self.photo_register_handler(self, user_id, photo_path)
            
            # 发送注册结果数据包
            self.send_register_result(success, user_id, msg)
            
            return success, msg
        except Exception as e:
            return False, str(e)
        finally:
            self.state = K230Protocol.STATE_IDLE
    
    def _handle_register_camera(self, params):
        """处理REGCAM命令（从摄像头注册）"""
        if len(params) < 1:
            return False, "Usage:REGCAM,user_id[,timeout]"
        
        user_id = params[0]
        timeout = int(params[1]) if len(params) > 1 else self.config.get('register_timeout', 10)
        
        if self.state == K230Protocol.STATE_RUNNING:
            return False, "BUSY,stop AI first"
        
        if self.state == K230Protocol.STATE_REGISTERING:
            return False, "BUSY,already registering"
        
        if self.camera_register_handler is None:
            return False, "Camera register not available"
        
        # 设置状态
        self.state = K230Protocol.STATE_REGISTERING
        self.send_response(K230Protocol.RSP_PROGRESS, "Registering from camera...")
        
        try:
            success, msg = self.camera_register_handler(self, user_id, timeout)
            
            # 发送注册结果数据包
            self.send_register_result(success, user_id, msg)
            
            return success, msg
        except Exception as e:
            return False, str(e)
        finally:
            self.state = K230Protocol.STATE_IDLE
    
    def _handle_list(self, params):
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
        if len(params) < 1:
            return False, "Usage:GET,key"
        
        key = params[0]
        if key in self.config:
            return True, "%s=%s" % (key, str(self.config[key]))
        else:
            return False, "Unknown key"
    
    def _handle_reload(self, params):
        """处理RELOAD命令（重新加载人脸数据库）"""
        return True, "Database will reload on next recognition"
    
    # ========== 命令处理循环 ==========
    def check_and_process_command(self):
        cmd_str = self.uart.receive_command()
        if cmd_str is None:
            return self.pending_stop
        
        cmd, params = self.protocol.parse_command(cmd_str)
        if cmd is None:
            return self.pending_stop
        
        print("[Controller] CMD:", cmd, params)
        
        # 运行状态下只处理特定命令
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
            elif cmd == K230Protocol.CMD_FRESET:
                # 强制重置可以在任何状态下执行
                self._handle_force_reset(params)
                self.send_response(K230Protocol.RSP_OK, "Force reset")
            else:
                self.send_response(K230Protocol.RSP_BUSY, "Running func:%d" % self.current_func_id)
            return self.pending_stop
        
        # 注册状态下只处理查询命令
        if self.state == K230Protocol.STATE_REGISTERING:
            if cmd == K230Protocol.CMD_STATUS:
                self.send_response(K230Protocol.RSP_OK, "%d,0" % self.state)
            elif cmd == K230Protocol.CMD_PING:
                self._handle_ping(params)
            else:
                self.send_response(K230Protocol.RSP_BUSY, "Registering")
            return False
        
        # 空闲状态处理所有命令
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
        if func_id in self.func_init_handlers:
            print("[Controller] Init function:", func_id)
            self.current_func_obj = self.func_init_handlers[func_id](self)
            return True
        return False
    
    def _run_function_once(self, func_id):
        if func_id in self.func_handlers:
            handler = self.func_handlers[func_id]
            handler(self, lambda: self.stop_flag)
    
    def _deinit_function(self, func_id):
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
        self.running = True
        self.uart.clear_buffer()
        
        print("[Controller] Running...")
        self.send_response(K230Protocol.RSP_READY, "K230")
        
        while self.running:
            try:
                if self.state == K230Protocol.STATE_IDLE:
                    self.check_and_process_command()
                    
                    if self.current_func_id != 0 and not self.stop_flag:
                        func_id = self.current_func_id
                        
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
                
                elif self.state == K230Protocol.STATE_RUNNING:
                    func_id = self.current_func_id
                    
                    try:
                        self._run_function_once(func_id)
                    except Exception as e:
                        print("[Controller] Run error:", e)
                        self.pending_stop = True
                    
                    self.check_and_process_command()
                    
                    if self.pending_stop or self.stop_flag:
                        print("[Controller] Stopping function:", func_id)
                        self._deinit_function(func_id)
                        self.state = K230Protocol.STATE_IDLE
                        self.current_func_id = 0
                        self.pending_stop = False
                        self.stop_flag = False
                        self.send_response(K230Protocol.RSP_OK, "Stopped")
                
                else:
                    self.check_and_process_command()
                    time.sleep_ms(10)
                
            except KeyboardInterrupt:
                print("[Controller] Keyboard interrupt")
                break
            except Exception as e:
                print("[Controller] Error:", e)
                time.sleep_ms(100)
        
        if self.state == K230Protocol.STATE_RUNNING:
            self._deinit_function(self.current_func_id)
        
        print("[Controller] Stopped")
    
    def stop(self):
        self.running = False
        self.pending_stop = True
        self.stop_flag = True
    
    def deinit(self):
        self.stop()
        self.uart.deinit()
        print("[Controller] Deinitialized")