"""
K230 串口通信模块
"""

from ybUtils.YbUart import YbUart

class K230Uart:
    """K230串口通信类"""
    
    def __init__(self, baudrate=115200, rx_buffer_size=512):
        self.uart = YbUart(baudrate=baudrate)
        self.rx_buffer = ""
        self.rx_buffer_max = rx_buffer_size
        self.debug = False
        print("[UART] Initialized at %d baud" % baudrate)
    
    def send(self, data):
        """发送数据"""
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
        接收命令（以 # 结尾的完整命令）
        支持 $CMD,...# 格式
        返回: 完整命令字符串 或 None
        """
        try:
            # 读取新数据
            data = self.uart.read()
            if data and len(data) > 0:
                try:
                    text = data.decode('utf-8')
                except:
                    text = data.decode('utf-8', 'ignore')
                
                # 清理换行符，统一处理
                text = text.replace('\r\n', '').replace('\r', '').replace('\n', '')
                self.rx_buffer += text
                
                # 防止缓冲区溢出
                if len(self.rx_buffer) > self.rx_buffer_max:
                    # 尝试找到最后一个 $ 开始保留
                    last_start = self.rx_buffer.rfind('$')
                    if last_start > 0:
                        self.rx_buffer = self.rx_buffer[last_start:]
                    else:
                        self.rx_buffer = self.rx_buffer[-256:]
            
            # 查找完整命令: $CMD,...#
            if '$' in self.rx_buffer and '#' in self.rx_buffer:
                start_idx = self.rx_buffer.find('$')
                end_idx = self.rx_buffer.find('#', start_idx)
                
                if start_idx >= 0 and end_idx > start_idx:
                    # 提取完整命令（包含 $ 和 #）
                    cmd = self.rx_buffer[start_idx:end_idx + 1]
                    # 移除已处理的部分
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
        """清空接收缓冲区"""
        self.rx_buffer = ""
        # 也清空硬件缓冲区
        try:
            while True:
                data = self.uart.read()
                if not data or len(data) == 0:
                    break
        except:
            pass
    
    def deinit(self):
        """释放资源"""
        print("[UART] Deinitialized")