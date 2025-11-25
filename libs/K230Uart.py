"""
K230 串口通信模块 - 多线程安全版本
"""

from ybUtils.YbUart import YbUart
import _thread

class K230Uart:
    """K230串口通信类 - 线程安全版本"""
    
    def __init__(self, baudrate=115200, rx_buffer_size=512):
        self.uart = YbUart(baudrate=baudrate)
        self.rx_buffer = ""
        self.rx_buffer_max = rx_buffer_size
        self.debug = False
        self.rx_lock = _thread.allocate_lock()  # 接收缓冲区锁
        print("[UART] Initialized at %d baud (thread-safe)" % baudrate)
    
    def send(self, data):
        """发送数据（调用方需确保线程安全）"""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')
            self.uart.send(data)
            if self.debug:
                print("[UART] TX:", data)
            return len(data)
        except Exception as e:
            print("[UART] Send error:", e)
            return 0
    
    def receive_command(self):
        """
        接收命令（线程安全）
        支持 $CMD,...# 格式
        返回: 完整命令字符串 或 None
        """
        self.rx_lock.acquire()
        try:
            return self._receive_command_internal()
        finally:
            self.rx_lock.release()
    
    def _receive_command_internal(self):
        """内部接收处理（需在锁保护下调用）"""
        try:
            # 读取新数据
            data = self.uart.read()
            if data and len(data) > 0:
                try:
                    text = data.decode('utf-8')
                except:
                    text = data.decode('utf-8', 'ignore')
                
                text = text.replace('\r\n', '').replace('\r', '').replace('\n', '')
                self.rx_buffer += text
                
                # 防止缓冲区溢出
                if len(self.rx_buffer) > self.rx_buffer_max:
                    last_start = self.rx_buffer.rfind('$')
                    if last_start > 0:
                        self.rx_buffer = self.rx_buffer[last_start:]
                    else:
                        self.rx_buffer = self.rx_buffer[-256:]
            
            # 查找完整命令
            if '$' in self.rx_buffer and '#' in self.rx_buffer:
                start_idx = self.rx_buffer.find('$')
                end_idx = self.rx_buffer.find('#', start_idx)
                
                if start_idx >= 0 and end_idx > start_idx:
                    cmd = self.rx_buffer[start_idx:end_idx + 1]
                    self.rx_buffer = self.rx_buffer[end_idx + 1:]
                    
                    if self.debug:
                        print("[UART] RX CMD:", cmd)
                    
                    return cmd
            
            return None
            
        except Exception as e:
            print("[UART] Receive error:", e)
            self.rx_buffer = ""
            return None
    
    def clear_buffer(self):
        """清空接收缓冲区（线程安全）"""
        self.rx_lock.acquire()
        try:
            self.rx_buffer = ""
            try:
                while True:
                    data = self.uart.read()
                    if not data or len(data) == 0:
                        break
            except:
                pass
        finally:
            self.rx_lock.release()
    
    def deinit(self):
        print("[UART] Deinitialized")