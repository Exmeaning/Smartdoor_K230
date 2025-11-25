"""
人脸识别功能模块 - 单帧处理模式
修复：使用 MediaHelper 正确释放资源
"""

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from libs.MediaHelper import MediaHelper, safe_cleanup
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import gc
import os
import math


class FaceDetApp(AIBase):
    """人脸检测子模块"""
    
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.5, nms_threshold=0.2,
                 rgb888p_size=[640, 480], display_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.anchors = anchors
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.ai2d.pad(self.get_pad_param(), 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            res = aidemo.face_det_post_process(
                self.confidence_threshold,
                self.nms_threshold,
                self.model_input_size[0],
                self.anchors,
                self.rgb888p_size,
                results
            )
            if len(res) == 0:
                return res, res
            else:
                return res[0], res[1]
    
    def get_pad_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        return [0, 0, 0, 0, int(round(0)), int(round(dh * 2 + 0.1)),
                int(round(0)), int(round(dw * 2 - 0.1))]


class FaceFeatureApp(AIBase):
    """人脸特征提取子模块"""
    
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640, 480],
                 display_size=[640, 480], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        self.debug_mode = debug_mode
        
        self.umeyama_args_112 = [
            38.2946, 51.6963, 73.5318, 51.5014, 56.0252, 71.7366,
            41.5493, 92.3655, 70.7299, 92.2041
        ]
        
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, landm, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
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
            matrix_dst = self.image_umeyama_112(sparse_points)
            return [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                    matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]


class FaceRecognitionEngine:
    """人脸识别引擎"""
    
    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.5, nms_threshold=0.2,
                 face_threshold=0.65, rgb888p_size=[640, 480], display_size=[640, 480]):
        
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.face_threshold = face_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        
        self.max_faces = 100
        self.feature_dim = 128
        self.db_names = []
        self.db_features = []
        
        self.face_det = FaceDetApp(
            self.face_det_kmodel,
            model_input_size=self.det_input_size,
            anchors=self.anchors,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            rgb888p_size=self.rgb888p_size,
            display_size=self.display_size
        )
        
        self.face_feature = FaceFeatureApp(
            self.face_reg_kmodel,
            model_input_size=self.reg_input_size,
            rgb888p_size=self.rgb888p_size,
            display_size=self.display_size
        )
        
        self.face_det.config_preprocess()
        self._load_database()
    
    def _load_database(self):
        try:
            try:
                os.stat(self.database_dir)
            except:
                print("[FaceRec] Database dir not found:", self.database_dir)
                return
            
            for item in os.listdir(self.database_dir):
                if len(self.db_names) >= self.max_faces:
                    break
                
                if item.endswith('.bin'):
                    filepath = self.database_dir + item
                    with open(filepath, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_features.append(feature)
                    self.db_names.append(item[:-4])
            
            print("[FaceRec] Loaded %d faces" % len(self.db_names))
            
        except Exception as e:
            print("[FaceRec] Load database error:", e)
    
    def run(self, input_np):
        det_boxes, landms = self.face_det.run(input_np)
        results = []
        
        for landm in landms:
            self.face_feature.config_preprocess(landm)
            feature = self.face_feature.run(input_np)
            name, score = self._search_database(feature)
            results.append((name, score))
        
        return det_boxes, results
    
    def _search_database(self, feature):
        if len(self.db_names) == 0:
            return "unknown", 0.0
        
        feature = feature / np.linalg.norm(feature)
        
        best_id = -1
        best_score = 0.0
        
        for i in range(len(self.db_names)):
            db_feature = self.db_features[i] / np.linalg.norm(self.db_features[i])
            score = np.dot(feature, db_feature) / 2 + 0.5
            
            if score > best_score:
                best_score = score
                best_id = i
        
        if best_id < 0 or best_score < self.face_threshold:
            return "unknown", best_score
        
        return self.db_names[best_id], best_score
    
    def reload_database(self):
        self.db_names = []
        self.db_features = []
        self._load_database()
    
    def get_models(self):
        """获取所有模型（用于统一释放）"""
        return [self.face_det, self.face_feature]
    
    def deinit(self):
        """释放模型资源（由 MediaHelper 调用）"""
        try:
            self.face_det.deinit()
        except:
            pass
        try:
            self.face_feature.deinit()
        except:
            pass


# ========== 功能接口 ==========

def face_recog_init(controller):
    """初始化人脸识别"""
    print("[FaceRecog] Initializing...")
    
    # 初始化前先强制清理
    MediaHelper.force_reset()
    
    kmodel_det = "/sdcard/kmodel/face_detection_320.kmodel"
    kmodel_reg = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((4200, 4))
    
    database_dir = controller.config.get('database_dir', '/data/face_database/')
    face_threshold = controller.config.get('face_threshold', 0.65)
    
    rgb888p_size = [640, 480]
    display_size = [640, 480]
    
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
    pl.create()
    MediaHelper.register_pipeline(pl)
    
    engine = FaceRecognitionEngine(
        kmodel_det, kmodel_reg,
        det_input_size=[320, 320],
        reg_input_size=[112, 112],
        database_dir=database_dir,
        anchors=anchors,
        face_threshold=face_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size
    )
    
    # 注册模型
    for model in engine.get_models():
        MediaHelper.register_model(model)
    
    print("[FaceRecog] Initialized")
    
    return {
        'pipeline': pl,
        'engine': engine,
        'display_size': display_size,
        'rgb888p_size': rgb888p_size,
        'frame_count': 0
    }


def face_recog_handler(controller, stop_check):
    """人脸识别处理函数 - 每次调用处理一帧"""
    obj = controller.current_func_obj
    if obj is None:
        return
    
    pl = obj['pipeline']
    engine = obj['engine']
    display_size = obj['display_size']
    rgb888p_size = obj['rgb888p_size']
    
    try:
        img = pl.get_frame()
        det_boxes, results = engine.run(img)
        
        pl.osd_img.clear()
        
        if det_boxes is not None and len(det_boxes) > 0:
            for i, det in enumerate(det_boxes):
                x, y, w, h = map(lambda v: int(round(v, 0)), det[:4])
                
                x_d = x * display_size[0] // rgb888p_size[0]
                y_d = y * display_size[1] // rgb888p_size[1]
                w_d = w * display_size[0] // rgb888p_size[0]
                h_d = h * display_size[1] // rgb888p_size[1]
                
                name, score = results[i] if i < len(results) else ("unknown", 0)
                
                if name == "unknown":
                    color = (255, 0, 0, 255)
                else:
                    color = (255, 0, 255, 0)
                
                pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, color=color, thickness=2)
                
                text = "%s:%.2f" % (name, score) if name != "unknown" else "unknown"
                pl.osd_img.draw_string_advanced(x_d, y_d - 20, 20, text, color=(255, 255, 0, 0))
                
                controller.send_face_recognition(x, y, w, h, name, score)
        
        pl.show_image()
        
        obj['frame_count'] = obj.get('frame_count', 0) + 1
        if obj['frame_count'] % 30 == 0:
            gc.collect()
        
    except Exception as e:
        print("[FaceRecog] Frame error:", e)


def face_recog_deinit(obj):
    """清理人脸识别 - 使用正确的清理顺序"""
    print("[FaceRecog] Deinitializing...")
    
    try:
        pipeline = obj.get('pipeline') if obj else None
        engine = obj.get('engine') if obj else None
        
        # 使用 MediaHelper 统一清理
        # 注意：engine 本身不是模型，需要获取其内部模型
        models = engine.get_models() if engine else None
        safe_cleanup(pipeline=pipeline, models=models)
        
    except Exception as e:
        print("[FaceRecog] Deinit error:", e)
        MediaHelper.force_reset()
    
    print("[FaceRecog] Deinitialized")