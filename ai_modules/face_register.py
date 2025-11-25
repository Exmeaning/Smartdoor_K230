"""
人脸注册功能模块
修复：摄像头模式下跳过模型deinit，避免阻塞
"""

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import image
import os
import gc
import math
import time


# 调试开关
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)


class FaceDetForReg(AIBase):
    """人脸检测类（用于注册）"""
    
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.5, nms_threshold=0.2,
                 rgb888p_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.debug_mode = debug_mode
        self.image_size = []
        
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("det set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.image_size = [ai2d_input_size[1], ai2d_input_size[0]]
            
            self.ai2d.pad(self.get_pad_param(ai2d_input_size), 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        with ScopedTiming("det postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(
                self.confidence_threshold,
                self.nms_threshold,
                self.model_input_size[0],
                self.anchors,
                self.image_size,
                results
            )
            if len(res) == 0:
                return None, None
            else:
                return res[0], res[1]
    
    def get_pad_param(self, image_input_size):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / image_input_size[0]
        ratio_h = dst_h / image_input_size[1]
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * image_input_size[0])
        new_h = int(ratio * image_input_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        return [0, 0, 0, 0, int(round(0)), int(round(dh * 2 + 0.1)),
                int(round(0)), int(round(dw * 2 - 0.1))]


class FaceFeatureForReg(AIBase):
    """人脸特征提取类（用于注册）"""
    
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.debug_mode = debug_mode
        
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]
        
        self.ai2d = None
        self._create_ai2d()
    
    def _create_ai2d(self):
        self.ai2d = Ai2d(self.debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, landm, input_image_size=None):
        with ScopedTiming("reg set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self._create_ai2d()
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        with ScopedTiming("reg postprocess", self.debug_mode > 0):
            return results[0][0]
    
    def svd22(self, a):
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]
        
        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) +
                math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))
        
        v[2] = (math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]),
                a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2)
                if s[0] > s[1] else 0)
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]
        
        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]
        
        v[0] = -v[0]
        v[2] = -v[2]
        return u, s, v
    
    def image_umeyama_112(self, src):
        SRC_NUM = 5
        SRC_DIM = 2
        
        src_mean = [0.0, 0.0]
        dst_mean = [0.0, 0.0]
        for i in range(0, SRC_NUM * 2, 2):
            src_mean[0] += src[i]
            src_mean[1] += src[i + 1]
            dst_mean[0] += self.umeyama_args_112[i]
            dst_mean[1] += self.umeyama_args_112[i + 1]
        src_mean[0] /= SRC_NUM
        src_mean[1] /= SRC_NUM
        dst_mean[0] /= SRC_NUM
        dst_mean[1] /= SRC_NUM
        
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]
        
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM
        
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]
        
        src_demean_mean = [0.0, 0.0]
        src_demean_var = [0.0, 0.0]
        for i in range(SRC_NUM):
            src_demean_mean[0] += src_demean[i][0]
            src_demean_mean[1] += src_demean[i][1]
        src_demean_mean[0] /= SRC_NUM
        src_demean_mean[1] /= SRC_NUM
        
        for i in range(SRC_NUM):
            src_demean_var[0] += (src_demean_mean[0] - src_demean[i][0]) ** 2
            src_demean_var[1] += (src_demean_mean[1] - src_demean[i][1]) ** 2
        src_demean_var[0] /= SRC_NUM
        src_demean_var[1] /= SRC_NUM
        
        scale = 1.0 / (src_demean_var[0] + src_demean_var[1]) * (S[0] + S[1])
        
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])
        
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale
        
        return T
    
    def get_affine_matrix(self, sparse_points):
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            if hasattr(sparse_points, 'tolist'):
                sparse_points = sparse_points.tolist()
            elif hasattr(sparse_points, '__iter__'):
                sparse_points = list(sparse_points)
            
            matrix_dst = self.image_umeyama_112(sparse_points)
            return [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                    matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]


class FaceRegister:
    """人脸注册器 - 修复摄像头模式下的资源释放问题"""
    
    def __init__(self, database_dir, debug_mode=1):
        self.database_dir = database_dir
        self.debug_mode = debug_mode
        
        self.face_det_kmodel = "/sdcard/kmodel/face_detection_320.kmodel"
        self.face_reg_kmodel = "/sdcard/kmodel/face_recognition.kmodel"
        self.anchors_path = "/sdcard/utils/prior_data_320.bin"
        
        self.det_input_size = [320, 320]
        self.reg_input_size = [112, 112]
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.2
        
        debug_print("Loading anchors...")
        self.anchors = np.fromfile(self.anchors_path, dtype=np.float)
        self.anchors = self.anchors.reshape((4200, 4))
        
        self.face_det = None
        self.face_reg = None
        
        self._ensure_dir(self.database_dir)
        print("[FaceRegister] Initialized, database:", self.database_dir)
    
    def _ensure_dir(self, directory):
        if not directory or directory == '/':
            return
        directory = directory.rstrip('/')
        try:
            os.stat(directory)
            return
        except OSError:
            if '/' in directory:
                parent = directory[:directory.rindex('/')]
                if parent and parent != directory:
                    self._ensure_dir(parent)
            try:
                os.mkdir(directory)
                print("[FaceRegister] Created dir:", directory)
            except:
                pass
    
    def _init_models(self, input_size):
        if self.face_det is not None:
            return
        
        print("[FaceRegister] Loading models...")
        
        self.face_det = FaceDetForReg(
            self.face_det_kmodel,
            model_input_size=self.det_input_size,
            anchors=self.anchors,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            rgb888p_size=input_size,
            debug_mode=self.debug_mode
        )
        
        self.face_reg = FaceFeatureForReg(
            self.face_reg_kmodel,
            model_input_size=self.reg_input_size,
            rgb888p_size=input_size,
            debug_mode=self.debug_mode
        )
        
        print("[FaceRegister] Models loaded")
    
    def _deinit_models_safe(self, use_camera=False):
        """安全释放模型
        
        Args:
            use_camera: 是否使用了摄像头模式。
                       如果是，不调用deinit()，只设置为None让GC处理
        """
        debug_print("=== Safe deinit models (camera=%s) ===" % use_camera)
        
        if use_camera:
            # 摄像头模式：不调用deinit，只清空引用
            # 因为在Pipeline存在时调用deinit会阻塞
            debug_print("Camera mode: skip deinit, set to None only")
            self.face_det = None
            self.face_reg = None
        else:
            # 静态图片模式：可以安全调用deinit
            if self.face_det:
                debug_print("Deinit face_det...")
                try:
                    self.face_det.deinit()
                except Exception as e:
                    debug_print("face_det deinit error:", e)
                self.face_det = None
            
            if self.face_reg:
                debug_print("Deinit face_reg...")
                try:
                    self.face_reg.deinit()
                except Exception as e:
                    debug_print("face_reg deinit error:", e)
                self.face_reg = None
        
        debug_print("Models released")
    
    def _image_to_nchw(self, img):
        img_rgb888 = img.to_rgb888()
        img_hwc = img_rgb888.to_numpy_ref()
        shape = img_hwc.shape
        
        img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
        img_trans = img_tmp.transpose()
        img_res = img_trans.copy()
        
        img_nchw = img_res.reshape((1, shape[2], shape[0], shape[1]))
        return img_nchw
    
    def _extract_feature(self, input_np, landm, input_size):
        debug_print("=== Starting feature extraction ===")
        debug_print("input_np shape:", input_np.shape)
        debug_print("input_size:", input_size)
        
        if hasattr(landm, 'tolist'):
            landm_list = landm.tolist()
        else:
            landm_list = list(landm)
        
        debug_print("landm_list:", landm_list[:10] if len(landm_list) >= 10 else landm_list)
        
        if len(landm_list) < 10:
            raise ValueError("landm should have at least 10 values")
        
        debug_print("Calling config_preprocess...")
        self.face_reg.config_preprocess(landm_list, input_image_size=input_size)
        debug_print("config_preprocess done")
        
        debug_print("Running feature extraction inference...")
        feature = self.face_reg.run(input_np)
        debug_print("Feature extracted, shape:", feature.shape if hasattr(feature, 'shape') else len(feature))
        
        return feature
    
    def register_from_photo(self, user_id, photo_path):
        """从照片注册人脸（不使用摄像头）"""
        print("[FaceRegister] Register from photo:", photo_path)
        
        try:
            try:
                os.stat(photo_path)
            except:
                return False, "Photo not found"
            
            img = image.Image(photo_path)
            input_np = self._image_to_nchw(img)
            input_size = [input_np.shape[3], input_np.shape[2]]
            
            self._init_models(input_size)
            self.face_det.config_preprocess(input_image_size=input_size)
            
            det_boxes, landms = self.face_det.run(input_np)
            
            if det_boxes is None or len(det_boxes) == 0:
                return False, "No face detected"
            
            if len(det_boxes) > 1:
                return False, "Multiple faces detected"
            
            feature = self._extract_feature(input_np, landms[0], input_size)
            
            feature_path = self.database_dir + user_id + ".bin"
            with open(feature_path, "wb") as f:
                f.write(feature.tobytes())
            
            print("[FaceRegister] Saved:", feature_path)
            return True, "Registered:" + user_id
            
        except Exception as e:
            import sys
            sys.print_exception(e)
            return False, str(e)
        
        finally:
            # 静态图片模式，可以安全deinit
            self._deinit_models_safe(use_camera=False)
            gc.collect()
    
    def register_from_camera(self, user_id, timeout_sec=10):
        """从摄像头注册人脸"""
        print("[FaceRegister] Register from camera, user:", user_id)
        
        pl = None
        registered = False
        message = "Timeout"
        
        try:
            rgb888p_size = [640, 480]
            display_size = [640, 480]
            
            debug_print("Creating pipeline...")
            pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
            pl.create()
            debug_print("Pipeline created")
            
            self._init_models(rgb888p_size)
            self.face_det.config_preprocess(input_image_size=rgb888p_size)
            
            start_time = time.time()
            stable_count = 0
            required_stable = 5
            last_landm = None
            
            print("[FaceRegister] Looking for face...")
            
            while time.time() - start_time < timeout_sec:
                img = pl.get_frame()
                det_boxes, landms = self.face_det.run(img)
                
                pl.osd_img.clear()
                
                if det_boxes is not None and len(det_boxes) == 1:
                    det = det_boxes[0]
                    x, y, w, h = map(lambda v: int(round(v, 0)), det[:4])
                    
                    face_area = w * h
                    frame_area = rgb888p_size[0] * rgb888p_size[1]
                    
                    if face_area > frame_area * 0.05:
                        stable_count += 1
                        last_landm = landms[0]
                        
                        x_d = x * display_size[0] // rgb888p_size[0]
                        y_d = y * display_size[1] // rgb888p_size[1]
                        w_d = w * display_size[0] // rgb888p_size[0]
                        h_d = h * display_size[1] // rgb888p_size[1]
                        
                        pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, 
                                                  color=(255, 0, 255, 0), thickness=4)
                        pl.osd_img.draw_string_advanced(x_d, y_d - 30, 24, 
                                                        "Hold... %d/%d" % (stable_count, required_stable),
                                                        color=(255, 255, 255, 0))
                        
                        if stable_count >= required_stable and last_landm is not None:
                            print("[FaceRegister] Face stable, extracting feature...")
                            
                            pl.osd_img.clear()
                            pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, 
                                                      color=(255, 255, 255, 0), thickness=4)
                            pl.osd_img.draw_string_advanced(x_d, y_d - 30, 24, 
                                                            "Processing...",
                                                            color=(255, 255, 0, 0))
                            pl.show_image()
                            
                            try:
                                debug_print("About to extract feature...")
                                feature = self._extract_feature(img, last_landm, rgb888p_size)
                                debug_print("Feature extracted successfully")
                                
                                feature_path = self.database_dir + user_id + ".bin"
                                debug_print("Saving to:", feature_path)
                                with open(feature_path, "wb") as f:
                                    f.write(feature.tobytes())
                                debug_print("Feature saved")
                                
                                registered = True
                                message = "Registered:" + user_id
                                
                                pl.osd_img.clear()
                                pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, 
                                                          color=(255, 0, 255, 0), thickness=4)
                                pl.osd_img.draw_string_advanced(x_d, y_d - 30, 24, 
                                                                "Success!",
                                                                color=(255, 0, 255, 0))
                                pl.show_image()
                                time.sleep(1)
                                break
                                
                            except Exception as e:
                                import sys
                                sys.print_exception(e)
                                stable_count = 0
                                last_landm = None
                                
                                pl.osd_img.clear()
                                pl.osd_img.draw_string_advanced(10, 10, 24, 
                                                                "Error, retry...",
                                                                color=(255, 255, 0, 0))
                    else:
                        stable_count = 0
                        last_landm = None
                        pl.osd_img.draw_string_advanced(10, 10, 24, 
                                                        "Move closer",
                                                        color=(255, 255, 0, 0))
                
                elif det_boxes is not None and len(det_boxes) > 1:
                    stable_count = 0
                    last_landm = None
                    pl.osd_img.draw_string_advanced(10, 10, 24, 
                                                    "One face only",
                                                    color=(255, 255, 0, 0))
                else:
                    stable_count = 0
                    last_landm = None
                    pl.osd_img.draw_string_advanced(10, 10, 24, 
                                                    "No face",
                                                    color=(255, 255, 0, 0))
                
                pl.show_image()
                gc.collect()
            
        except Exception as e:
            import sys
            sys.print_exception(e)
            print("[FaceRegister] Camera error:", e)
            message = str(e)
        
        # ========== 关键：正确的清理顺序 ==========
        debug_print("=== Cleanup phase ===")
        
        # 步骤1：清空模型引用（不调用deinit，避免阻塞）
        debug_print("Step 1: Release model references (no deinit)...")
        self._deinit_models_safe(use_camera=True)
        
        # 步骤2：销毁Pipeline
        if pl:
            debug_print("Step 2: Destroying pipeline...")
            try:
                pl.destroy()
                debug_print("Pipeline destroyed")
            except Exception as e:
                debug_print("Pipeline destroy error:", e)
            pl = None
        
        # 步骤3：强制GC
        debug_print("Step 3: Force gc.collect()...")
        gc.collect()
        debug_print("=== Cleanup completed ===")
        
        return registered, message
    
    def delete_user(self, user_id):
        try:
            feature_path = self.database_dir + user_id + ".bin"
            os.remove(feature_path)
            return True, "Deleted:" + user_id
        except:
            return False, "User not found"
    
    def list_users(self):
        try:
            users = []
            for item in os.listdir(self.database_dir):
                if item.endswith('.bin'):
                    users.append(item[:-4])
            return users
        except:
            return []


# ========== 命令处理函数 ==========

def face_register_from_photo(controller, user_id, photo_path):
    database_dir = controller.config.get('database_dir', '/data/face_database/')
    registrar = FaceRegister(database_dir, debug_mode=1)
    return registrar.register_from_photo(user_id, photo_path)


def face_register_from_camera(controller, user_id, timeout=10):
    database_dir = controller.config.get('database_dir', '/data/face_database/')
    registrar = FaceRegister(database_dir, debug_mode=1)
    return registrar.register_from_camera(user_id, timeout)


def face_register_handler(controller, user_id, photo_path):
    return face_register_from_photo(controller, user_id, photo_path)