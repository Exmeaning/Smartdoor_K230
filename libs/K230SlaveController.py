"""
K230 从机控制器 - 多线程版本
使用 _thread 实现命令处理和AI运行分离
"""

import _thread
import time
import gc
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart

class K230SlaveController:
    """K230从机主控制器 - 多线程版本"""
    
    def __init__(self, baudrate=115200):
        self.uart = K230Uart(baudrate=baudrate)
        self.protocol = K230Protocol()
        
        # ========== 状态变量 ==========
        self.state = K230Protocol.STATE_IDLE
        self.current_func_id = 0
        self.running = False
        
        # ========== 线程控制 ==========
        self.lock = _thread.allocate_lock()          # 状态锁
        self.uart_lock = _thread.allocate_lock()     # 串口发送锁
        self.ai_thread_running = False               # AI线程运行标志
        self.ai_stop_request = False                 # AI停止请求
        self.ai_start_request = False                # AI启动请求
        self.pending_func_id = 0                     # 待启动的功能ID
        
        # ========== 功能注册 ==========
        self.func_handlers = {}
        self.func_init_handlers = {}
        self.func_deinit_handlers = {}
        self.func_once_handlers = {}
        
        # ========== 命令处理器 ==========
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
        
        # ========== 配置 ==========
        self.config = {
            'face_threshold': 0.65,
            'detect_threshold': 0.5,
            'nms_threshold': 0.2,
            'database_dir': '/data/face_database/',
            'register_timeout': 10,
        }
        
        # ========== 功能对象 ==========
        self.current_func_obj = None
        
        # ========== 注册处理器 ==========
        self.photo_register_handler = None
        self.camera_register_handler = None
        
        print("[Controller] Initialized (Multi-thread mode)")
    
    # ========== 线程安全的状态操作 ==========
    def _get_state(self):
        """线程安全获取状态"""
        self.lock.acquire()
        try:
            return self.state
        finally:
            self.lock.release()
    
    def _set_state(self, new_state):
        """线程安全设置状态"""
        self.lock.acquire()
        try:
            self.state = new_state
        finally:
            self.lock.release()
    
    def _check_stop_flag(self):
        """检查是否需要停止（给AI线程用）"""
        self.lock.acquire()
        try:
            return self.ai_stop_request
        finally:
            self.lock.release()
    
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
        self.photo_register_handler = handler
        print("[Controller] Registered photo register handler")
    
    def register_camera_handler(self, handler):
        self.camera_register_handler = handler
        print("[Controller] Registered camera register handler")
    
    # ========== 线程安全的数据发送 ==========
    def send_response(self, rsp_type, msg="", data=None):
        rsp = self.protocol.build_response(rsp_type, msg, data)
        self._send_safe(rsp)
    
    def _send_safe(self, data):
        """线程安全发送"""
        self.uart_lock.acquire()
        try:
            self.uart.send(data)
        finally:
            self.uart_lock.release()
    
    def send_face_detect(self, x, y, w, h):
        packet = self.protocol.get_face_detect_data(x, y, w, h)
        self._send_safe(packet)
    
    def send_face_recognition(self, x, y, w, h, name, score):
        packet = self.protocol.get_face_recognition_data(x, y, w, h, name, score)
        self._send_safe(packet)
    
    def send_person_detect(self, x, y, w, h):
        packet = self.protocol.get_person_detect_data(x, y, w, h)
        self._send_safe(packet)
    
    def send_object_detect(self, x, y, w, h, label):
        packet = self.protocol.get_object_detect_data(x, y, w, h, label)
        self._send_safe(packet)
    
    def send_register_result(self, success, user_id, msg=""):
        packet = self.protocol.get_register_result_data(success, user_id, msg)
        self._send_safe(packet)
    
    # ========== AI线程函数 ==========
    def _ai_thread_func(self, func_id):
        """AI处理线程主函数"""
        print("[AI Thread] Started for func:", func_id)
        
        self.lock.acquire()
        self.ai_thread_running = True
        self.ai_stop_request = False
        self.lock.release()
        
        try:
            # 初始化功能
            if func_id in self.func_init_handlers:
                print("[AI Thread] Initializing...")
                self.current_func_obj = self.func_init_handlers[func_id](self)
            
            self._set_state(K230Protocol.STATE_RUNNING)
            self.send_response(K230Protocol.RSP_OK, "Started:%d" % func_id)
            
            # 获取处理函数
            handler = self.func_handlers.get(func_id)
            if handler is None:
                raise Exception("No handler for func %d" % func_id)
            
            frame_count = 0
            
            # 主处理循环
            while not self._check_stop_flag():
                try:
                    # 执行一帧处理
                    handler(self, self._check_stop_flag)
                    frame_count += 1
                    
                    # 定期GC
                    if frame_count % 50 == 0:
                        gc.collect()
                    
                    # 【关键】让出CPU时间，让主线程有机会处理命令
                    time.sleep_us(1)
                    
                except Exception as e:
                    print("[AI Thread] Frame error:", e)
                    time.sleep_ms(10)
            
            print("[AI Thread] Stop requested, cleaning up...")
            
        except Exception as e:
            print("[AI Thread] Error:", e)
            import sys
            sys.print_exception(e)
        
        finally:
            # 清理功能
            try:
                if func_id in self.func_deinit_handlers and self.current_func_obj:
                    print("[AI Thread] Deinitializing...")
                    self.func_deinit_handlers[func_id](self.current_func_obj)
            except Exception as e:
                print("[AI Thread] Deinit error:", e)
            
            self.current_func_obj = None
            
            # 更新状态
            self.lock.acquire()
            self.ai_thread_running = False
            self.current_func_id = 0
            self.lock.release()
            
            self._set_state(K230Protocol.STATE_IDLE)
            self.send_response(K230Protocol.RSP_OK, "Stopped")
            
            gc.collect()
            print("[AI Thread] Exited")
    
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
        
        current_state = self._get_state()
        
        if current_state == K230Protocol.STATE_RUNNING:
            return False, "BUSY,stop first"
        
        if current_state == K230Protocol.STATE_REGISTERING:
            return False, "BUSY,registering"
        
        # 检查线程是否真正停止
        self.lock.acquire()
        if self.ai_thread_running:
            self.lock.release()
            return False, "BUSY,thread running"
        self.lock.release()
        
        # 启动AI线程
        self.current_func_id = func_id
        
        try:
            _thread.start_new_thread(self._ai_thread_func, (func_id,))
            return True, "Starting:%d" % func_id
        except Exception as e:
            return False, "Thread error:%s" % str(e)
    
    def _handle_stop(self, params):
        current_state = self._get_state()
        
        if current_state == K230Protocol.STATE_IDLE:
            return True, "Already stopped"
        
        if current_state == K230Protocol.STATE_REGISTERING:
            return False, "Cannot stop registration"
        
        # 设置停止标志
        self.lock.acquire()
        self.ai_stop_request = True
        self.lock.release()
        
        return True, "Stopping"
    
    def _handle_status(self, params):
        self.lock.acquire()
        status_str = "%d,%d" % (self.state, self.current_func_id)
        self.lock.release()
        return True, status_str
    
    def _handle_ping(self, params):
        self.send_response(K230Protocol.RSP_PONG, "K230")
        return None, None
    
    def _handle_reset(self, params):
        # 请求停止AI线程
        self.lock.acquire()
        self.ai_stop_request = True
        self.lock.release()
        
        # 等待线程停止
        timeout = 30  # 3秒超时
        while timeout > 0:
            self.lock.acquire()
            running = self.ai_thread_running
            self.lock.release()
            
            if not running:
                break
            
            time.sleep_ms(100)
            timeout -= 1
        
        gc.collect()
        return True, "Reset"
    
    def _handle_force_reset(self, params):
        """处理 FRESET 命令"""
        try:
            from libs.MediaHelper import MediaHelper
            
            # 请求停止AI线程
            self.lock.acquire()
            self.ai_stop_request = True
            self.lock.release()
            
            # 等待一会
            time.sleep_ms(500)
            
            # 强制重置
            self._set_state(K230Protocol.STATE_IDLE)
            self.current_func_id = 0
            self.current_func_obj = None
            
            MediaHelper.force_reset()
            
            return True, "Force reset done"
        except Exception as e:
            return False, str(e)
    
    def _handle_register_photo(self, params):
        """处理REG命令"""
        if len(params) < 2:
            return False, "Usage:REG,user_id,photo_path"
        
        user_id = params[0]
        photo_path = params[1]
        
        current_state = self._get_state()
        if current_state != K230Protocol.STATE_IDLE:
            return False, "BUSY"
        
        if self.photo_register_handler is None:
            return False, "Photo register not available"
        
        self._set_state(K230Protocol.STATE_REGISTERING)
        self.send_response(K230Protocol.RSP_PROGRESS, "Registering from photo...")
        
        try:
            success, msg = self.photo_register_handler(self, user_id, photo_path)
            self.send_register_result(success, user_id, msg)
            return success, msg
        except Exception as e:
            return False, str(e)
        finally:
            self._set_state(K230Protocol.STATE_IDLE)
    
    def _handle_register_camera(self, params):
        """处理REGCAM命令"""
        if len(params) < 1:
            return False, "Usage:REGCAM,user_id[,timeout]"
        
        user_id = params[0]
        timeout = int(params[1]) if len(params) > 1 else self.config.get('register_timeout', 10)
        
        current_state = self._get_state()
        if current_state != K230Protocol.STATE_IDLE:
            return False, "BUSY"
        
        # 检查AI线程
        self.lock.acquire()
        if self.ai_thread_running:
            self.lock.release()
            return False, "BUSY,AI running"
        self.lock.release()
        
        if self.camera_register_handler is None:
            return False, "Camera register not available"
        
        self._set_state(K230Protocol.STATE_REGISTERING)
        self.send_response(K230Protocol.RSP_PROGRESS, "Registering from camera...")
        
        try:
            success, msg = self.camera_register_handler(self, user_id, timeout)
            self.send_register_result(success, user_id, msg)
            return success, msg
        except Exception as e:
            return False, str(e)
        finally:
            self._set_state(K230Protocol.STATE_IDLE)
    
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
        return True, "Database will reload on next recognition"
    
    # ========== 主线程命令处理循环 ==========
    def _process_commands(self):
        """处理接收到的命令"""
        cmd_str = self.uart.receive_command()
        if cmd_str is None:
            return
        
        cmd, params = self.protocol.parse_command(cmd_str)
        if cmd is None:
            return
        
        print("[Controller] CMD:", cmd, params)
        
        current_state = self._get_state()
        
        # 运行状态下只处理特定命令
        if current_state == K230Protocol.STATE_RUNNING:
            if cmd == K230Protocol.CMD_STOP:
                result = self._handle_stop(params)
                self.send_response(K230Protocol.RSP_OK, result[1])
            elif cmd == K230Protocol.CMD_STATUS:
                result = self._handle_status(params)
                self.send_response(K230Protocol.RSP_OK, result[1])
            elif cmd == K230Protocol.CMD_PING:
                self._handle_ping(params)
            elif cmd == K230Protocol.CMD_FRESET:
                result = self._handle_force_reset(params)
                if result[0]:
                    self.send_response(K230Protocol.RSP_OK, result[1])
                else:
                    self.send_response(K230Protocol.RSP_ERROR, result[1])
            else:
                self.send_response(K230Protocol.RSP_BUSY, "Running func:%d" % self.current_func_id)
            return
        
        # 注册状态
        if current_state == K230Protocol.STATE_REGISTERING:
            if cmd == K230Protocol.CMD_STATUS:
                self.send_response(K230Protocol.RSP_OK, "%d,0" % current_state)
            elif cmd == K230Protocol.CMD_PING:
                self._handle_ping(params)
            else:
                self.send_response(K230Protocol.RSP_BUSY, "Registering")
            return
        
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
    
    # ========== 主循环 ==========
    def run(self):
        """主线程运行（只负责命令处理）"""
        self.running = True
        self.uart.clear_buffer()
        
        print("[Controller] Running (Multi-thread mode)...")
        self.send_response(K230Protocol.RSP_READY, "K230")
        
        while self.running:
            try:
                # 处理命令
                self._process_commands()
                
                # 【关键】主线程也要让出CPU
                time.sleep_us(1)
                
                # 空闲时稍微休息一下，节省资源
                current_state = self._get_state()
                if current_state == K230Protocol.STATE_IDLE:
                    time.sleep_ms(5)
                
            except KeyboardInterrupt:
                print("[Controller] Keyboard interrupt")
                break
            except Exception as e:
                print("[Controller] Error:", e)
                time.sleep_ms(100)
        
        # 清理
        self.lock.acquire()
        self.ai_stop_request = True
        self.lock.release()
        
        # 等待AI线程停止
        timeout = 50
        while timeout > 0:
            self.lock.acquire()
            running = self.ai_thread_running
            self.lock.release()
            if not running:
                break
            time.sleep_ms(100)
            timeout -= 1
        
        print("[Controller] Stopped")
    
    def stop(self):
        self.running = False
        self.lock.acquire()
        self.ai_stop_request = True
        self.lock.release()
    
    def deinit(self):
        self.stop()
        time.sleep_ms(500)
        self.uart.deinit()
        print("[Controller] Deinitialized")