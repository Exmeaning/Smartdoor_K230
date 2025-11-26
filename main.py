"""
K230 从机端主程序 - 延迟加载版
基于 DeepWiki 分析的正确架构
"""

from libs.PipeLine import PipeLine, ScopedTiming
from libs.K230Protocol import K230Protocol
from libs.K230Uart import K230Uart
import gc
import time

# ========== 配置 ==========
RGB888P_SIZE = [640, 480]
DISPLAY_SIZE = [640, 480]
DISPLAY_MODE = "lcd"
UART_BAUDRATE = 115200

class RunMode:
    IDLE = 0
    FACE_DETECTION = 6
    FACE_RECOGNITION = 8
    FACE_REGISTRATION = 100


class K230Controller:
    """K230 控制器 - 延迟加载架构"""
    
    def __init__(self):
        # 串口通信
        self.uart = K230Uart(baudrate=UART_BAUDRATE)
        self.protocol = K230Protocol()
        
        # Pipeline（核心资源管理器）
        self.pl = None
        
        # 当前模式
        self.current_mode = RunMode.IDLE
        
        # 功能模块（延迟初始化）
        self.face_det_module = None
        self.face_rec_module = None
        self.face_reg_module = None
        
        # 当前活动模块
        self.active_module = None
        
        # 运行标志
        self.running = False
        
        # 配置
        self.config = {
            'database_dir': '/data/face_database/',
            'face_threshold': 0.65,
        }
        
        print("[Controller] Created")
    
    def init(self):
        """初始化系统（只初始化 PipeLine，不加载模型）"""
        print("[Controller] Initializing...")
        
        # 创建 Pipeline
        self.pl = PipeLine(
            rgb888p_size=RGB888P_SIZE,
            display_size=DISPLAY_SIZE,
            display_mode=DISPLAY_MODE
        )
        self.pl.create()
        
        print("[Controller] Pipeline created")
        self._send_response(K230Protocol.RSP_READY, "K230")
    
    # ========== 模块管理（延迟加载）==========
    
    def _get_or_create_module(self, mode):
        """获取或创建模块（不初始化）"""
        if mode == RunMode.FACE_DETECTION:
            if self.face_det_module is None:
                from modules.face_detection import FaceDetectionModule
                self.face_det_module = FaceDetectionModule(
                    self.uart, self.protocol,
                    RGB888P_SIZE, DISPLAY_SIZE
                )
            return self.face_det_module
        
        elif mode == RunMode.FACE_RECOGNITION:
            if self.face_rec_module is None:
                from modules.face_recognition import FaceRecognitionModule
                self.face_rec_module = FaceRecognitionModule(
                    self.uart, self.protocol,
                    RGB888P_SIZE, DISPLAY_SIZE,
                    self.config['database_dir'],
                    self.config['face_threshold']
                )
            return self.face_rec_module
        
        elif mode == RunMode.FACE_REGISTRATION:
            if self.face_reg_module is None:
                from modules.face_registration import FaceRegistrationModule
                self.face_reg_module = FaceRegistrationModule(
                    self.uart, self.protocol,
                    RGB888P_SIZE, DISPLAY_SIZE,
                    self.config['database_dir']
                )
            return self.face_reg_module
        
        return None
    
    def _stop_current_module(self):
        """停止并释放当前模块"""
        if self.active_module:
            print("[Controller] Stopping current module...")
            self.active_module.stop()
            self.active_module.deinit()
            self.active_module = None
            gc.collect()
            print("[Controller] Module stopped")
    
    # ========== 命令处理 ==========
    
    def _handle_start(self, params):
        """处理 START 命令"""
        if len(params) < 1:
            return False, "Missing func_id"
        
        try:
            func_id = int(params[0])
        except:
            return False, "Invalid func_id"
        
        print("[Controller] START func_id=%d" % func_id)
        
        # 停止当前模块
        self._stop_current_module()
        
        # 获取目标模块
        module = self._get_or_create_module(func_id)
        if module is None:
            return False, "Unknown function"
        
        # 初始化并启动（此时才加载模型！）
        module.init()
        module.start()
        
        self.active_module = module
        self.current_mode = func_id
        
        return True, "Started:%d" % func_id
    
    def _handle_stop(self, params):
        """处理 STOP 命令"""
        self._stop_current_module()
        self.current_mode = RunMode.IDLE
        return True, "Stopped"
    
    def _handle_register(self, params):
        """处理人脸注册命令"""
        if len(params) < 1:
            return False, "Missing user_id"
        
        user_id = params[0]
        
        # 获取注册模块
        module = self._get_or_create_module(RunMode.FACE_REGISTRATION)
        if module is None:
            return False, "Module error"
        
        # 【关键】暂停当前模块，但不销毁
        was_running = self.active_module is not None
        if was_running:
            self.active_module.stop()
        
        # 执行注册（复用 Pipeline）
        success, msg = module.register_from_camera(self.pl, user_id)
        
        # 恢复之前的模块
        if was_running and self.active_module:
            self.active_module.start()
        
        # 如果识别模块存在，重新加载数据库
        if success and self.face_rec_module:
            self.face_rec_module.reload_database()
        
        return success, msg
    
    def _handle_ping(self, params):
        self._send_response(K230Protocol.RSP_PONG, "K230")
        return None, None
    
    def _handle_status(self, params):
        return True, "%d,%d" % (
            1 if self.active_module else 0,
            self.current_mode
        )
    
    def _process_command(self):
        """处理串口命令"""
        cmd_str = self.uart.receive_command()
        if cmd_str is None:
            return
        
        cmd, params = self.protocol.parse_command(cmd_str)
        if cmd is None:
            return
        
        print("[CMD]", cmd, params)
        
        handlers = {
            K230Protocol.CMD_START: self._handle_start,
            K230Protocol.CMD_STOP: self._handle_stop,
            K230Protocol.CMD_PING: self._handle_ping,
            K230Protocol.CMD_STATUS: self._handle_status,
            K230Protocol.CMD_REGCAM: self._handle_register,
        }
        
        if cmd in handlers:
            result = handlers[cmd](params)
            if result and result[0] is not None:
                success, msg = result
                rsp = K230Protocol.RSP_OK if success else K230Protocol.RSP_ERROR
                self._send_response(rsp, msg)
        else:
            self._send_response(K230Protocol.RSP_ERROR, "Unknown:" + cmd)
    
    def _send_response(self, rsp_type, msg):
        rsp = self.protocol.build_response(rsp_type, msg)
        self.uart.send(rsp)
    
    # ========== 主循环 ==========
    
    def run(self):
        """主循环"""
        self.init()
        self.running = True
        
        print("[Controller] ========== Running ==========")
        
        frame_count = 0
        
        while self.running:
            try:
                # 处理命令
                self._process_command()
                
                # 处理当前模块
                if self.active_module and self.active_module.is_running():
                    self.active_module.run_once(self.pl)
                else:
                    # 空闲时也要刷新显示
                    self.pl.get_frame()
                    self.pl.osd_img.clear()
                    self.pl.show_image()
                
                # 【修复】更频繁的 GC（人脸识别消耗大）
                frame_count += 1
                if frame_count % 50 == 0:
                    gc.collect()
                
                time.sleep_ms(5)  # 稍微增加延迟，减轻系统压力
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print("[Controller] Error:", e)
                import sys
                sys.print_exception(e)
                time.sleep_ms(100)
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        print("[Controller] Cleaning up...")
        
        self._stop_current_module()
        
        # 释放所有模块
        for module in [self.face_det_module, self.face_rec_module, self.face_reg_module]:
            if module:
                try:
                    module.deinit()
                except:
                    pass
        
        # 销毁 Pipeline
        if self.pl:
            self.pl.destroy()
        
        gc.collect()
        print("[Controller] Cleanup complete")
    
    def stop(self):
        self.running = False


def main():
    print("=" * 50)
    print("K230 Slave Controller - Lazy Loading")
    print("=" * 50)
    
    controller = K230Controller()
    
    try:
        controller.run()
    except Exception as e:
        print("Error:", e)
        import sys
        sys.print_exception(e)
    finally:
        controller.cleanup()


if __name__ == "__main__":
    main()