"""
K230 串口通信模块
基于已有的YbUart封装
"""

from ybUtils.YbUart import YbUart

class K230Uart:
    """K230串口通信类 - 封装YbUart"""
    
    def __init__(self, baudrate=115200, rx_buffer_size=512):
        """
        初始化串口
        
        参数:
            baudrate: 波特率
            rx_buffer_size: 接收缓冲区大小
        """
        # 使用已有的YbUart类（已配置好FPIOA）
        self.uart = YbUart(baudrate=baudrate)
        
        self.rx_buffer = ""
        self.rx_buffer_max = rx_buffer_size
        self.debug = False
        
        print("[UART] Initialized at %d baud" % baudrate)
    
    def send(self, data):
        """
        发送数据
        
        参数:
            data: 字符串或字节数据
        """
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
    
    def send_line(self, text):
        """发送一行文本（自动添加换行）"""
        if not text.endswith('\n'):
            text += '\n'
        return self.send(text)
    
    def receive_line(self):
        """
        接收一行数据（非阻塞）
        
        返回:
            收到完整行返回字符串，否则返回None
        """
        try:
            # 使用YbUart的receive方法读取数据
            data = self.uart.read()
            if data and len(data) > 0:
                # 解码并添加到缓冲区
                try:
                    self.rx_buffer += data.decode('utf-8')
                except:
                    self.rx_buffer += data.decode('utf-8', 'ignore')
                
                # 防止缓冲区溢出
                if len(self.rx_buffer) > self.rx_buffer_max:
                    self.rx_buffer = self.rx_buffer[-self.rx_buffer_max:]
            
            # 查找完整的行
            if '\n' in self.rx_buffer:
                idx = self.rx_buffer.index('\n')
                line = self.rx_buffer[:idx].strip()
                self.rx_buffer = self.rx_buffer[idx + 1:]
                
                if self.debug and line:
                    print("[UART] RX:", line)
                
                return line if line else None
            
            return None
            
        except Exception as e:
            print("[UART] Receive error:", e)
            self.rx_buffer = ""
            return None
    
    def clear_buffer(self):
        """清空接收缓冲区"""
        self.rx_buffer = ""
    
    def deinit(self):
        """释放资源"""
        print("[UART] Deinitialized")