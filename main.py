"""
K230 从机端主程序 - 摄像头常驻版
"""

from libs.K230SlaveController import K230SlaveController
from libs.K230Protocol import K230Protocol

def main():
    print("=" * 50)
    print("K230 Slave Controller")
    print("Camera Always-On Mode")
    print("=" * 50)
    
    controller = K230SlaveController(baudrate=115200)
    
    # 可选：注册摄像头注册处理器
    # from ai_modules.face_register import face_register_from_camera
    # controller.register_camera_handler(face_register_from_camera)
    
    print("\n=== Commands ===")
    print("  $CMD,START,6#     - Face Detection")
    print("  $CMD,START,8#     - Face Recognition")
    print("  $CMD,STOP#        - Stop AI")
    print("  $CMD,SET,fps,15#  - Set camera FPS")
    print("  $CMD,GET,fps#     - Get camera FPS")
    print("  $CMD,PING#        - Test connection")
    print("  $CMD,STATUS#      - Get status")
    print("=" * 50 + "\n")
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
    except Exception as e:
        print("Error:", e)
        import sys
        sys.print_exception(e)
    finally:
        controller.deinit()

if __name__ == "__main__":
    main()