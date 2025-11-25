"""
K230 从机端主程序
单线程状态机模式
"""

import time
import gc
from libs.K230SlaveController import K230SlaveController
from libs.K230Protocol import K230Protocol

from ai_modules.face_detection import (
    face_detect_init,
    face_detect_handler,
    face_detect_deinit
)

from ai_modules.face_recognition import (
    face_recog_init,
    face_recog_handler,
    face_recog_deinit
)

from ai_modules.face_register import face_register_handler


def main():
    print("=" * 50)
    print("K230 Slave Controller")
    print("Single-thread State Machine Mode")
    print("=" * 50)
    
    controller = K230SlaveController(baudrate=115200)
    
    # 注册循环执行的功能
    controller.register_function(
        K230Protocol.ID_FACE_DETECT,
        face_detect_handler,
        face_detect_init,
        face_detect_deinit
    )
    
    controller.register_function(
        K230Protocol.ID_FACE_RECOGNITION,
        face_recog_handler,
        face_recog_init,
        face_recog_deinit
    )
    
    # 注册一次性执行的功能
    controller.register_once_function(
        K230Protocol.ID_FACE_REGISTER,
        face_register_handler
    )
    
    # 设置配置
    controller.config['database_dir'] = '/data/face_database/'
    controller.config['face_threshold'] = 0.65
    controller.config['detect_threshold'] = 0.5
    
    print("\nRegistered functions:")
    print("  - Face Detection (ID=%d)" % K230Protocol.ID_FACE_DETECT)
    print("  - Face Recognition (ID=%d)" % K230Protocol.ID_FACE_RECOGNITION)
    print("  - Face Register (ID=%d)" % K230Protocol.ID_FACE_REGISTER)
    print("\nCommands:")
    print("  $CMD,PING#           - Test connection")
    print("  $CMD,START,6#        - Start face detection")
    print("  $CMD,START,8#        - Start face recognition")
    print("  $CMD,STOP#           - Stop current function")
    print("  $CMD,STATUS#         - Get status")
    print("=" * 50)
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print("Fatal error:", e)
    finally:
        controller.deinit()
        print("Stopped")


if __name__ == "__main__":
    main()