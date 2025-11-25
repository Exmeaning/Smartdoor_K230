"""
人脸注册模块 - 复用 PipeLine 版
"""

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import os
import gc
import time
import math

# 配置
FACE_DET_KMODEL = "/sdcard/kmodel/face_detection_320.kmodel"
FACE_REG_KMODEL = "/sdcard/kmodel/face_recognition.kmodel"
ANCHORS_PATH = "/sdcard/utils/prior_data_320.bin"

FACE_DET_INPUT_SIZE = [320, 320]
FACE_REG_INPUT_SIZE = [112, 112]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2


class FaceRegistrationModule:
    """人脸注册模块"""
    
    def __init__(self, uart, protocol, rgb888p_size, display_size, 
                 database_dir, debug_mode=0):
        self.uart = uart
        self.protocol = protocol
        self.rgb888p_size = rgb888p_size
        self.display_size = display_size
        self.database_dir = database_dir
        self.debug_mode = debug_mode
        
        self.face_det = None
        self.face_reg = None
        self.anchors = None
        
        self.initialized = False
        self.running = False
        
        # 确保目录存在
        self._ensure_dir(database_dir)
    
    def _ensure_dir(self, path):
        path = path.rstrip('/')
        try:
            os.stat(path)
        except:
            parent = path[:path.rfind('/')]
            if parent:
                self._ensure_dir(parent)
            try:
                os.mkdir(path)
            except:
                pass
    
    def init(self):
        """初始化模型"""
        if self.initialized:
            return
        
        print("[FaceReg] Loading models...")
        
        # 加载 anchors
        self.anchors = np.fromfile(ANCHORS_PATH, dtype=np.float)
        self.anchors = self.anchors.reshape((4200, 4))
        
        # 导入并创建检测器（复用 face_recognition 的类）
        from modules.face_recognition import FaceDetApp, FaceFeatureApp
        
        self.face_det = FaceDetApp(
            kmodel_path=FACE_DET_KMODEL,
            model_input_size=FACE_DET_INPUT_SIZE,
            anchors=self.anchors,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            rgb888p_size=self.rgb888p_size,
            debug_mode=self.debug_mode
        )
        self.face_det.config_preprocess()
        
        self.face_reg = FaceFeatureApp(
            kmodel_path=FACE_REG_KMODEL,
            model_input_size=FACE_REG_INPUT_SIZE,
            rgb888p_size=self.rgb888p_size,
            debug_mode=self.debug_mode
        )
        
        self.initialized = True
        print("[FaceReg] Models loaded")
    
    def start(self):
        self.running = True
    
    def stop(self):
        self.running = False
    
    def is_running(self):
        return self.running
    
    def run_once(self, pl):
        """注册模块不需要在主循环中运行"""
        pass
    
    def register_from_camera(self, pl, user_id, timeout_sec=10):
        """
        从摄像头注册人脸
        【关键】复用传入的 PipeLine，不创建新的
        """
        print("[FaceReg] Registering user:", user_id)
        
        # 确保模型已初始化
        if not self.initialized:
            self.init()
        
        registered = False
        message = "Timeout"
        
        start_time = time.time()
        stable_count = 0
        required_stable = 5
        last_landm = None
        
        try:
            while time.time() - start_time < timeout_sec:
                # 获取帧（使用传入的 PipeLine）
                img = pl.get_frame()
                
                # 检测
                det_boxes, landms = self.face_det.run(img)
                
                # 清除 OSD
                pl.osd_img.clear()
                
                if det_boxes is not None and len(det_boxes) == 1:
                    det = det_boxes[0]
                    x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
                    
                    # 检查人脸大小
                    face_area = w * h
                    frame_area = self.rgb888p_size[0] * self.rgb888p_size[1]
                    
                    if face_area > frame_area * 0.05:
                        stable_count += 1
                        last_landm = landms[0]
                        
                        # 坐标转换
                        x_d = x * self.display_size[0] // self.rgb888p_size[0]
                        y_d = y * self.display_size[1] // self.rgb888p_size[1]
                        w_d = w * self.display_size[0] // self.rgb888p_size[0]
                        h_d = h * self.display_size[1] // self.rgb888p_size[1]
                        
                        # 绘制进度
                        pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d,
                                                   color=(255, 0, 255, 0), thickness=4)
                        pl.osd_img.draw_string_advanced(
                            x_d, y_d - 30, 24,
                            "Hold... %d/%d" % (stable_count, required_stable),
                            color=(255, 255, 255, 0)
                        )
                        
                        # 稳定帧数足够，提取特征
                        if stable_count >= required_stable and last_landm is not None:
                            pl.osd_img.draw_string_advanced(
                                x_d, y_d - 30, 24, "Processing...",
                                color=(255, 255, 0, 0)
                            )
                            pl.show_image()
                            
                            try:
                                # 配置并提取特征
                                self.face_reg.config_preprocess(last_landm)
                                feature = self.face_reg.run(img)
                                
                                # 保存特征
                                feature_path = self.database_dir + user_id + ".bin"
                                with open(feature_path, "wb") as f:
                                    f.write(feature.tobytes())
                                
                                registered = True
                                message = "Registered:" + user_id
                                
                                # 显示成功
                                pl.osd_img.clear()
                                pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d,
                                                           color=(255, 0, 255, 0), thickness=4)
                                pl.osd_img.draw_string_advanced(
                                    x_d, y_d - 30, 24, "Success!",
                                    color=(255, 0, 255, 0)
                                )
                                pl.show_image()
                                time.sleep(1)
                                break
                                
                            except Exception as e:
                                print("[FaceReg] Feature error:", e)
                                stable_count = 0
                                last_landm = None
                    else:
                        stable_count = 0
                        last_landm = None
                        pl.osd_img.draw_string_advanced(
                            10, 10, 24, "Move closer",
                            color=(255, 255, 0, 0)
                        )
                
                elif det_boxes is not None and len(det_boxes) > 1:
                    stable_count = 0
                    last_landm = None
                    pl.osd_img.draw_string_advanced(
                        10, 10, 24, "One face only",
                        color=(255, 255, 0, 0)
                    )
                else:
                    stable_count = 0
                    last_landm = None
                    pl.osd_img.draw_string_advanced(
                        10, 10, 24, "No face",
                        color=(255, 255, 0, 0)
                    )
                
                pl.show_image()
                gc.collect()
        
        except Exception as e:
            print("[FaceReg] Error:", e)
            import sys
            sys.print_exception(e)
            message = str(e)
        
        return registered, message
    
    def deinit(self):
        """释放资源"""
        print("[FaceReg] Deinitializing...")
        
        if self.face_det:
            self.face_det.deinit()
            self.face_det = None
        
        if self.face_reg:
            self.face_reg.deinit()
            self.face_reg = None
        
        self.anchors = None
        self.initialized = False
        self.running = False
        
        gc.collect()
        print("[FaceReg] Deinitialized")