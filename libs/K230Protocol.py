"""
K230 通信协议模块
支持双向通信的协议解析与构建
"""

class K230Protocol:
    """K230通信协议处理类"""
    
    # ========== 功能ID定义 ==========
    ID_COLOR = 1
    ID_BARCODE = 2
    ID_QRCODE = 3
    ID_APRILTAG = 4
    ID_DMCODE = 5
    ID_FACE_DETECT = 6
    ID_EYE_GAZE = 7
    ID_FACE_RECOGNITION = 8
    ID_PERSON_DETECT = 9
    ID_FALLDOWN_DETECT = 10
    ID_HAND_DETECT = 11
    ID_HAND_GESTURE = 12
    ID_OCR_REC = 13
    ID_OBJECT_DETECT = 14
    ID_NANO_TRACKER = 15
    ID_SELF_LEARNING = 16
    ID_LICENCE_REC = 17
    ID_LICENCE_DETECT = 18
    ID_GARBAGE_DETECT = 19
    ID_GUIDE_DETECT = 20
    ID_OBSTACLE_DETECT = 21
    ID_MULTI_COLOR = 22
    ID_FINGER_GUESS = 23
    ID_FACE_REGISTER = 100  # 自定义: 人脸注册
    
    # ========== 命令类型定义 ==========
    CMD_START = "START"       # 启动功能
    CMD_STOP = "STOP"         # 停止功能
    CMD_REG = "REG"           # 人脸注册
    CMD_STATUS = "STATUS"     # 查询状态
    CMD_RESET = "RESET"       # 重置系统
    CMD_PING = "PING"         # 心跳检测
    CMD_LIST = "LIST"         # 列出注册人脸
    CMD_DELETE = "DELETE"     # 删除注册人脸
    CMD_SET = "SET"           # 设置参数
    CMD_GET = "GET"           # 获取参数
    
    # ========== 响应类型定义 ==========
    RSP_OK = "OK"
    RSP_ERROR = "ERR"
    RSP_BUSY = "BUSY"
    RSP_READY = "READY"
    RSP_PONG = "PONG"
    RSP_DATA = "DATA"
    
    # ========== 状态定义 ==========
    STATE_IDLE = 0
    STATE_RUNNING = 1
    STATE_ERROR = 2
    STATE_BUSY = 3
    
    def __init__(self):
        self.debug = False
    
    # ========== 命令解析 ==========
    def parse_command(self, data):
        """
        解析接收到的命令
        命令格式: $CMD,<cmd_type>,<param1>,<param2>,...#
        
        返回: (cmd_type, params_list) 或 (None, None)
        """
        try:
            data = data.strip()
            
            # 检查协议头尾
            if not data.startswith('$CMD,') or not data.endswith('#'):
                if self.debug:
                    print("[Protocol] Invalid format:", data)
                return None, None
            
            # 提取内容 (去掉 $CMD, 和 #)
            content = data[5:-1]
            parts = content.split(',')
            
            if len(parts) < 1:
                return None, None
            
            cmd_type = parts[0].upper()
            params = parts[1:] if len(parts) > 1 else []
            
            if self.debug:
                print("[Protocol] Parsed:", cmd_type, params)
            
            return cmd_type, params
            
        except Exception as e:
            print("[Protocol] Parse error:", e)
            return None, None
    
    # ========== 响应构建 ==========
    def build_response(self, rsp_type, msg="", extra_data=None):
        """
        构建响应包
        响应格式: $RSP,<len>,<rsp_type>,<msg>[,<extra_data>]#\n
        """
        if extra_data is None:
            temp = "$RSP,00,%s,%s#" % (rsp_type, msg)
            pto_len = len(temp)
            return "$RSP,%02d,%s,%s#\n" % (pto_len, rsp_type, msg)
        else:
            temp = "$RSP,00,%s,%s,%s#" % (rsp_type, msg, extra_data)
            pto_len = len(temp)
            return "$RSP,%02d,%s,%s,%s#\n" % (pto_len, rsp_type, msg, extra_data)
    
    # ========== 数据包构建方法 ==========
    def build_coord_packet(self, func_id, x, y, w, h, msg=None):
        """构建坐标数据包"""
        if msg is None:
            temp = "$%02d,%02d,%03d,%03d,%03d,%03d#" % (0, func_id, x, y, w, h)
            pto_len = len(temp)
            return "$%02d,%02d,%03d,%03d,%03d,%03d#\n" % (pto_len, func_id, x, y, w, h)
        else:
            temp = "$%02d,%02d,%03d,%03d,%03d,%03d,%s#" % (0, func_id, x, y, w, h, msg)
            pto_len = len(temp)
            return "$%02d,%02d,%03d,%03d,%03d,%03d,%s#\n" % (pto_len, func_id, x, y, w, h, msg)
    
    def build_msg_value_packet(self, func_id, x, y, w, h, msg, value):
        """构建带消息和值的数据包"""
        temp = "$%02d,%02d,%03d,%03d,%03d,%03d,%s,%03d#" % (0, func_id, x, y, w, h, msg, value)
        pto_len = len(temp)
        return "$%02d,%02d,%03d,%03d,%03d,%03d,%s,%03d#\n" % (pto_len, func_id, x, y, w, h, msg, value)
    
    def build_message_packet(self, func_id, msg, value=None):
        """构建消息数据包"""
        if value is None:
            temp = "$%02d,%02d,%s#" % (0, func_id, msg)
            pto_len = len(temp)
            return "$%02d,%02d,%s#\n" % (pto_len, func_id, msg)
        else:
            temp = "$%02d,%02d,%s,%03d#" % (0, func_id, msg, value)
            pto_len = len(temp)
            return "$%02d,%02d,%s,%03d#\n" % (pto_len, func_id, msg, value)
    
    # ========== 特定功能数据包 ==========
    def get_face_detect_data(self, x, y, w, h):
        """人脸检测数据包"""
        return self.build_coord_packet(self.ID_FACE_DETECT, int(x), int(y), int(w), int(h))
    
    def get_face_recognition_data(self, x, y, w, h, name, score):
        """人脸识别数据包"""
        return self.build_msg_value_packet(
            self.ID_FACE_RECOGNITION, 
            int(x), int(y), int(w), int(h), 
            name, 
            int(float(score) * 100)
        )
    
    def get_person_detect_data(self, x, y, w, h):
        """人体检测数据包"""
        return self.build_coord_packet(self.ID_PERSON_DETECT, int(x), int(y), int(w), int(h))
    
    def get_hand_detect_data(self, x, y, w, h):
        """手部检测数据包"""
        return self.build_coord_packet(self.ID_HAND_DETECT, int(x), int(y), int(w), int(h))
    
    def get_hand_gesture_data(self, gesture):
        """手势识别数据包"""
        return self.build_message_packet(self.ID_HAND_GESTURE, gesture)
    
    def get_object_detect_data(self, x, y, w, h, label):
        """物体检测数据包"""
        return self.build_coord_packet(self.ID_OBJECT_DETECT, int(x), int(y), int(w), int(h), label)
    
    def get_register_result_data(self, success, user_id, msg=""):
        """人脸注册结果数据包"""
        status = 1 if success else 0
        return self.build_message_packet(self.ID_FACE_REGISTER, "%s,%d,%s" % (user_id, status, msg))