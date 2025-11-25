"""
K230 从机端主程序
支持人脸检测、识别、注册功能
"""

import time
import gc
from libs.K230SlaveController import K230SlaveController
from libs.K230Protocol import K230Protocol

# 导入功能模块
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

from ai_modules.face_register import (
    face_register_from_photo,
    face_register_from_camera
)


def main():
    print("=" * 50)
    print("K230 Slave Controller")
    print("Face Detection / Recognition / Registration")
    print("=" * 50)
    
    # 创建控制器
    controller = K230SlaveController(baudrate=115200)
    
    # 注册循环执行的AI功能
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
    
    # 注册人脸注册处理器
    controller.register_photo_handler(face_register_from_photo)
    controller.register_camera_handler(face_register_from_camera)
    
    # 设置配置
    controller.config['database_dir'] = '/data/face_database/'
    controller.config['face_threshold'] = 0.65
    controller.config['detect_threshold'] = 0.5
    controller.config['register_timeout'] = 15
    
    # 打印帮助信息
    print("\n=== Registered Functions ===")
    print("  Face Detection   (ID=%d)" % K230Protocol.ID_FACE_DETECT)
    print("  Face Recognition (ID=%d)" % K230Protocol.ID_FACE_RECOGNITION)
    
    print("\n=== Available Commands ===")
    print("  $CMD,PING#                      - Test connection")
    print("  $CMD,STATUS#                    - Get status")
    print("  $CMD,START,6#                   - Start face detection")
    print("  $CMD,START,8#                   - Start face recognition")
    print("  $CMD,STOP#                      - Stop current AI function")
    print("  $CMD,REG,user_id,/path/photo#   - Register from photo")
    print("  $CMD,REGCAM,user_id#            - Register from camera")
    print("  $CMD,REGCAM,user_id,15#         - Register with 15s timeout")
    print("  $CMD,LIST#                      - List registered users")
    print("  $CMD,DELETE,user_id#            - Delete user")
    print("  $CMD,SET,key,value#             - Set config")
    print("  $CMD,GET,key#                   - Get config")
    print("  $CMD,RESET#                     - Reset system")
    
    print("\n=== Config Keys ===")
    print("  database_dir     - Face database path")
    print("  face_threshold   - Recognition threshold (0.0-1.0)")
    print("  detect_threshold - Detection threshold (0.0-1.0)")
    print("  register_timeout - Camera register timeout (seconds)")
    
    print("\n" + "=" * 50)
    print("Waiting for commands...")
    print("=" * 50 + "\n")
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print("Fatal error:", e)
    finally:
        controller.deinit()
        print("K230 Slave Controller stopped")


if __name__ == "__main__":
    main()