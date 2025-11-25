"""
K230 AI处理器 - 修复版
解决与 CameraManager 的资源冲突
"""

from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import gc
import time
import os

# 调试开关
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print("[AI-DBG]", *args)


class AIProcessor:
    """AI处理器 - 修复版"""
    
    MODEL_NONE = 0
    MODEL_FACE_DETECT = 1
    MODEL_FACE_RECOGNIZE = 2
    
    def __init__(self, rgb_size=[640, 480], display_size=[640, 480]):
        self.rgb_size = [self._align_up(rgb_size[0], 16), rgb_size[1]]
        self.display_size = [self._align_up(display_size[0], 16), display_size[1]]
        
        self.current_model_type = self.MODEL_NONE
        self._models = {}
        
        # 模型路径
        self.kmodel_paths = {
            'face_det': "/sdcard/kmodel/face_detection_320.kmodel",
            'face_reg': "/sdcard/kmodel/face_recognition.kmodel",
        }
        self.anchors_path = "/sdcard/utils/prior_data_320.bin"
        
        # 预加载 anchors（这个是纯数据，不涉及硬件资源）
        self._anchors = None
        self._preload_anchors()
        
        # 人脸数据库
        self.database_dir = "/data/face_database/"
        self.db_names = []
        self.db_features = []
        self.face_threshold = 0.65
        
        # 检测参数
        self.det_threshold = 0.5
        self.nms_threshold = 0.2
        
        print("[AI] Processor created (deferred loading)")
    
    def _align_up(self, value, align):
        return ((value + align - 1) // align) * align
    
    def _preload_anchors(self):
        """预加载 anchors 数据"""
        try:
            debug_print("Preloading anchors...")
            os.stat(self.anchors_path)
            self._anchors = np.fromfile(self.anchors_path, dtype=np.float)
            self._anchors = self._anchors.reshape((4200, 4))
            debug_print("Anchors loaded, shape:", self._anchors.shape)
        except Exception as e:
            print("[AI] Warning: anchors preload failed:", e)
            self._anchors = None
    
    def update_size(self, rgb_size, display_size):
        """更新尺寸（在摄像头启动后调用）"""
        self.rgb_size = [self._align_up(rgb_size[0], 16), rgb_size[1]]
        self.display_size = [self._align_up(display_size[0], 16), display_size[1]]
        debug_print("Size updated:", self.rgb_size, self.display_size)
    
    def load(self, model_type, **kwargs):
        """加载指定类型的模型"""
        if self.current_model_type == model_type:
            debug_print("Model already loaded:", model_type)
            return True
        
        if self.current_model_type != self.MODEL_NONE:
            self.unload()
        
        debug_print("Loading model type:", model_type)
        
        # 更新参数
        self.det_threshold = kwargs.get('det_threshold', 0.5)
        self.nms_threshold = kwargs.get('nms_threshold', 0.2)
        self.face_threshold = kwargs.get('face_threshold', 0.65)
        self.database_dir = kwargs.get('database_dir', '/data/face_database/')
        
        try:
            # 强制GC
            gc.collect()
            time.sleep_ms(100)
            
            if model_type == self.MODEL_FACE_DETECT:
                self._load_face_detect()
            elif model_type == self.MODEL_FACE_RECOGNIZE:
                self._load_face_recognize()
            else:
                print("[AI] Unknown model type:", model_type)
                return False
            
            self.current_model_type = model_type
            gc.collect()
            print("[AI] Model loaded successfully")
            return True
            
        except Exception as e:
            print("[AI] Load error:", e)
            import sys
            sys.print_exception(e)
            self.unload()
            return False
    
    def unload(self):
        """卸载当前模型"""
        if self.current_model_type == self.MODEL_NONE:
            return
        
        debug_print("Unloading models...")
        
        for name, model in self._models.items():
            try:
                if model and hasattr(model, 'deinit'):
                    debug_print("  deinit:", name)
                    model.deinit()
            except Exception as e:
                debug_print("  deinit error:", name, e)
        
        self._models.clear()
        self.current_model_type = self.MODEL_NONE
        
        gc.collect()
        debug_print("Models unloaded")
    
    def is_loaded(self):
        return self.current_model_type != self.MODEL_NONE
    
    def get_model_type(self):
        return self.current_model_type
    
    # ========== 人脸检测 ==========
    
    def _load_face_detect(self):
        """加载人脸检测模型"""
        debug_print("=== Loading face detection ===")
        
        # 检查 anchors
        if self._anchors is None:
            self._preload_anchors()
        
        if self._anchors is None:
            raise RuntimeError("Anchors not available")
        
        # 检查 kmodel 文件
        kmodel_path = self.kmodel_paths['face_det']
        debug_print("Checking kmodel:", kmodel_path)
        try:
            stat = os.stat(kmodel_path)
            debug_print("kmodel size:", stat[6], "bytes")
        except:
            raise FileNotFoundError("kmodel not found: " + kmodel_path)
        
        # 创建检测器
        debug_print("Creating FaceDetector...")
        debug_print("  rgb_size:", self.rgb_size)
        debug_print("  model_input_size: [320, 320]")
        
        det = FaceDetector(
            kmodel_path=kmodel_path,
            model_input_size=[320, 320],
            anchors=self._anchors,
            confidence_threshold=self.det_threshold,
            nms_threshold=self.nms_threshold,
            rgb888p_size=self.rgb_size
        )
        debug_print("FaceDetector created")
        
        debug_print("Calling config_preprocess...")
        det.config_preprocess()
        debug_print("config_preprocess done")
        
        self._models['face_det'] = det
        debug_print("=== Face detection loaded ===")
    
    def _process_face_detect(self, frame):
        """处理人脸检测"""
        det = self._models.get('face_det')
        if det is None:
            return []
        
        try:
            dets = det.run(frame)
            if dets is None or len(dets) == 0:
                return []
            
            results = []
            for d in dets:
                x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                results.append((x, y, w, h))
            
            return results
            
        except Exception as e:
            print("[AI] Detect error:", e)
            return []
    
    # ========== 人脸识别 ==========
    
    def _load_face_recognize(self):
        """加载人脸识别模型"""
        debug_print("=== Loading face recognition ===")
        
        # 先加载检测模型
        self._load_face_detect()
        
        # 检查 kmodel 文件
        kmodel_path = self.kmodel_paths['face_reg']
        debug_print("Checking kmodel:", kmodel_path)
        try:
            os.stat(kmodel_path)
        except:
            raise FileNotFoundError("kmodel not found: " + kmodel_path)
        
        # 加载特征提取模型
        debug_print("Creating FaceFeature...")
        reg = FaceFeature(
            kmodel_path=kmodel_path,
            model_input_size=[112, 112],
            rgb888p_size=self.rgb_size
        )
        debug_print("FaceFeature created")
        
        self._models['face_reg'] = reg
        
        # 加载人脸数据库
        self._load_database()
        debug_print("=== Face recognition loaded ===")
    
    def _load_database(self):
        """加载人脸特征数据库"""
        self.db_names = []
        self.db_features = []
        
        try:
            try:
                os.stat(self.database_dir)
            except:
                debug_print("Database dir not found, creating...")
                self._ensure_dir(self.database_dir)
                return
            
            for item in os.listdir(self.database_dir):
                if item.endswith('.bin'):
                    filepath = self.database_dir + item
                    with open(filepath, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_features.append(feature)
                    self.db_names.append(item[:-4])
            
            print("[AI] Loaded %d faces from database" % len(self.db_names))
            
        except Exception as e:
            print("[AI] Load database error:", e)
    
    def _ensure_dir(self, path):
        """确保目录存在"""
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
    
    def reload_database(self):
        """重新加载人脸数据库"""
        self._load_database()
    
    def _process_face_recognize(self, frame):
        """处理人脸识别"""
        det = self._models.get('face_det')
        reg = self._models.get('face_reg')
        
        if det is None or reg is None:
            return []
        
        try:
            det_result = det.run(frame)
            if det_result is None:
                return []
            
            det_boxes, landms = det_result if isinstance(det_result, tuple) else (det_result, [])
            
            if len(det_boxes) == 0:
                return []
            
            results = []
            
            for i, d in enumerate(det_boxes):
                x, y, w, h = int(d[0]), int(d[1]), int(d[2]), int(d[3])
                
                name = "unknown"
                score = 0.0
                
                if i < len(landms):
                    try:
                        reg.config_preprocess(landms[i])
                        feature = reg.run(frame)
                        name, score = self._match_feature(feature)
                    except Exception as e:
                        debug_print("Feature extraction error:", e)
                
                results.append((x, y, w, h, name, score))
            
            return results
            
        except Exception as e:
            print("[AI] Recognize error:", e)
            return []
    
    def _match_feature(self, feature):
        """匹配特征"""
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
            return "unknown", float(best_score)
        
        return self.db_names[best_id], float(best_score)
    
    def process(self, frame):
        """处理一帧"""
        if frame is None:
            return []
        
        if self.current_model_type == self.MODEL_FACE_DETECT:
            return self._process_face_detect(frame)
        elif self.current_model_type == self.MODEL_FACE_RECOGNIZE:
            return self._process_face_recognize(frame)
        else:
            return []


# ========== 内部模型类 ==========

class FaceDetector(AIBase):
    """人脸检测器"""
    
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold, nms_threshold, rgb888p_size):
        debug_print("FaceDetector.__init__ start")
        debug_print("  kmodel_path:", kmodel_path)
        debug_print("  model_input_size:", model_input_size)
        debug_print("  rgb888p_size:", rgb888p_size)
        
        # 调用父类构造函数（这里可能卡住）
        debug_print("  Calling AIBase.__init__...")
        super().__init__(kmodel_path, model_input_size, rgb888p_size, 0)
        debug_print("  AIBase.__init__ done")
        
        self.model_input_size = model_input_size
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = rgb888p_size
        
        debug_print("  Creating Ai2d...")
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
        debug_print("FaceDetector.__init__ done")
    
    def config_preprocess(self, input_size=None):
        debug_print("FaceDetector.config_preprocess start")
        size = input_size if input_size else self.rgb888p_size
        top, bottom, left, right = self._get_padding()
        
        debug_print("  padding:", top, bottom, left, right)
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        
        debug_print("  building ai2d...")
        self.ai2d.build([1, 3, size[1], size[0]],
                       [1, 3, self.model_input_size[1], self.model_input_size[0]])
        debug_print("FaceDetector.config_preprocess done")
    
    def postprocess(self, results):
        res = aidemo.face_det_post_process(
            self.confidence_threshold,
            self.nms_threshold,
            self.model_input_size[0],
            self.anchors,
            self.rgb888p_size,
            results
        )
        if len(res) == 0:
            return None
        return (res[0], res[1]) if len(res) > 1 else res[0]
    
    def _get_padding(self):
        dst_w, dst_h = self.model_input_size
        ratio = min(dst_w / self.rgb888p_size[0], dst_h / self.rgb888p_size[1])
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        return (0, int(round(dh * 2 + 0.1)), 0, int(round(dw * 2 - 0.1)))


class FaceFeature(AIBase):
    """人脸特征提取"""
    
    def __init__(self, kmodel_path, model_input_size, rgb888p_size):
        debug_print("FaceFeature.__init__ start")
        super().__init__(kmodel_path, model_input_size, rgb888p_size, 0)
        debug_print("FaceFeature.__init__ AIBase done")
        
        self.model_input_size = model_input_size
        self.rgb888p_size = rgb888p_size
        
        self.umeyama_args = [38.2946, 51.6963, 73.5318, 51.5014, 56.0252,
                             71.7366, 41.5493, 92.3655, 70.7299, 92.2041]
        
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
        debug_print("FaceFeature.__init__ done")
    
    def config_preprocess(self, landm, input_size=None):
        size = input_size if input_size else self.rgb888p_size
        
        self.ai2d = Ai2d(0)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
        
        matrix = self._get_affine_matrix(landm)
        self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, matrix)
        self.ai2d.build([1, 3, size[1], size[0]],
                       [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        return results[0][0]
    
    def _get_affine_matrix(self, src):
        import math
        
        if hasattr(src, 'tolist'):
            src = src.tolist()
        
        src_mean = [sum(src[0::2])/5, sum(src[1::2])/5]
        dst_mean = [sum(self.umeyama_args[0::2])/5, sum(self.umeyama_args[1::2])/5]
        
        src_demean = [[src[2*i] - src_mean[0], src[2*i+1] - src_mean[1]] for i in range(5)]
        dst_demean = [[self.umeyama_args[2*i] - dst_mean[0], 
                       self.umeyama_args[2*i+1] - dst_mean[1]] for i in range(5)]
        
        A = [[0, 0], [0, 0]]
        for i in range(2):
            for k in range(2):
                for j in range(5):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= 5
        
        a = [A[0][0], A[0][1], A[1][0], A[1][1]]
        s0 = (math.sqrt((a[0]-a[3])**2 + (a[1]+a[2])**2) + 
              math.sqrt((a[0]+a[3])**2 + (a[1]-a[2])**2)) / 2
        s1 = abs(s0 - math.sqrt((a[0]-a[3])**2 + (a[1]+a[2])**2))
        
        v2 = math.sin(math.atan2(2*(a[0]*a[1]+a[2]*a[3]), 
                                  a[0]**2-a[1]**2+a[2]**2-a[3]**2)/2) if s0 > s1 else 0
        v0 = math.sqrt(1 - v2**2)
        
        u0 = -(a[0]*v0 + a[1]*v2) / s0 if s0 != 0 else 1
        u2 = -(a[2]*v0 + a[3]*v2) / s0 if s0 != 0 else 0
        u1 = (a[0]*(-v2) + a[1]*v0) / s1 if s1 != 0 else -u2
        u3 = (a[2]*(-v2) + a[3]*v0) / s1 if s1 != 0 else u0
        
        T00 = u0*(-v0) + u1*v2
        T01 = u0*(-v2) + u1*(-v0)
        T10 = u2*(-v0) + u3*v2
        T11 = u2*(-v2) + u3*(-v0)
        
        var = sum([(src_demean[i][0]**2 + src_demean[i][1]**2)/5 for i in range(5)])
        scale = (s0 + s1) / var if var > 0 else 1
        
        T02 = dst_mean[0] - scale * (T00*src_mean[0] + T01*src_mean[1])
        T12 = dst_mean[1] - scale * (T10*src_mean[0] + T11*src_mean[1])
        
        return [T00*scale, T01*scale, T02, T10*scale, T11*scale, T12]