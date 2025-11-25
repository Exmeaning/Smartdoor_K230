"""
K230 从机端主程序
启动从机控制器，等待树莓派主机的命令
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

from ai_modules.face_register import face_register_handler


def main():
    """主函数"""
    print("=" * 50)
    print("K230 Slave Controller Starting...")
    print("=" * 50)
    
    # 创建控制器 (使用默认波特率115200，YbUart会处理引脚配置)
    controller = K230SlaveController(baudrate=115200)
    
    # 注册人脸检测功能
    controller.register_function(
        K230Protocol.ID_FACE_DETECT,
        face_detect_handler,
        face_detect_init,
        face_detect_deinit
    )
    
    # 注册人脸识别功能
    controller.register_function(
        K230Protocol.ID_FACE_RECOGNITION,
        face_recog_handler,
        face_recog_init,
        face_recog_deinit
    )
    
    # 注册人脸注册功能（特殊处理，不是循环执行的）
    controller.register_function(
        K230Protocol.ID_FACE_REGISTER,
        face_register_handler
    )
    
    # 设置默认配置
    controller.config['database_dir'] = '/data/face_database/'
    controller.config['face_threshold'] = 0.65
    controller.config['detect_threshold'] = 0.5
    
    print("Registered functions:")
    print("  - Face Detection (ID=%d)" % K230Protocol.ID_FACE_DETECT)
    print("  - Face Recognition (ID=%d)" % K230Protocol.ID_FACE_RECOGNITION)
    print("  - Face Register (ID=%d)" % K230Protocol.ID_FACE_REGISTER)
    print("")
    print("Waiting for commands from master...")
    print("=" * 50)
    
    try:
        # 运行主循环
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