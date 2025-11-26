"""
人脸识别模块 - 延迟加载版
"""

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import os
import gc
import math

# 配置
FACE_DET_KMODEL = "/sdcard/kmodel/face_detection_320.kmodel"
FACE_REG_KMODEL = "/sdcard/kmodel/face_recognition.kmodel"
ANCHORS_PATH = "/sdcard/utils/prior_data_320.bin"

FACE_DET_INPUT_SIZE = [320, 320]
FACE_REG_INPUT_SIZE = [112, 112]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2


class FaceDetApp(AIBase):
    """人脸检测（用于识别）"""
    
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold, nms_threshold, rgb888p_size, debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.model_input_size = model_input_size
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = rgb888p_size
        
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, input_image_size=None):
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        top, bottom, left, right = self._get_padding(ai2d_input_size)
        
        self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
        self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
        self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                       [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
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
            return None, None
        return res[0], res[1]
    
    def _get_padding(self, size):
        dst_w, dst_h = self.model_input_size
        ratio = min(dst_w / size[0], dst_h / size[1])
        new_w, new_h = int(ratio * size[0]), int(ratio * size[1])
        dw, dh = (dst_w - new_w) / 2, (dst_h - new_h) / 2
        return (0, int(round(dh * 2 + 0.1)), 0, int(round(dw * 2 - 0.1)))


class FaceFeatureApp(AIBase):
    """人脸特征提取"""
    
    def __init__(self, kmodel_path, model_input_size, rgb888p_size, debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.model_input_size = model_input_size
        self.rgb888p_size = rgb888p_size
        
        # 标准人脸关键点
        self.std_points = [38.2946, 51.6963, 73.5318, 51.5014, 56.0252,
                          71.7366, 41.5493, 92.3655, 70.7299, 92.2041]
        
        self.ai2d = None  # 延迟创建
    
    def config_preprocess(self, landm, input_image_size=None):
        """【关键】每次检测到人脸都需要重新配置"""
        ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
        
        # 【修复】先释放旧的 ai2d 资源
        if self.ai2d is not None:
            try:
                del self.ai2d
                self.ai2d = None
            except:
                pass
            gc.collect()  # 强制回收
        
        # 重建 ai2d
        self.ai2d = Ai2d(self.debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
        
        # 计算仿射变换矩阵
        matrix = self._get_affine_matrix(landm)
        self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, matrix)
        self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                       [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        return results[0][0]
    
    def _get_affine_matrix(self, src):
        """计算仿射变换矩阵（Umeyama 算法）"""
        if hasattr(src, 'tolist'):
            src = src.tolist()
        
        # 计算均值
        src_mean = [sum(src[0::2])/5, sum(src[1::2])/5]
        dst_mean = [sum(self.std_points[0::2])/5, sum(self.std_points[1::2])/5]
        
        # 去均值
        src_demean = [[src[2*i] - src_mean[0], src[2*i+1] - src_mean[1]] for i in range(5)]
        dst_demean = [[self.std_points[2*i] - dst_mean[0], 
                       self.std_points[2*i+1] - dst_mean[1]] for i in range(5)]
        
        # 协方差矩阵
        A = [[0, 0], [0, 0]]
        for i in range(2):
            for k in range(2):
                for j in range(5):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= 5
        
        # SVD 分解
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
        
        T00, T01 = u0*(-v0) + u1*v2, u0*(-v2) + u1*(-v0)
        T10, T11 = u2*(-v0) + u3*v2, u2*(-v2) + u3*(-v0)
        
        var = sum([(src_demean[i][0]**2 + src_demean[i][1]**2)/5 for i in range(5)])
        scale = (s0 + s1) / var if var > 0 else 1
        
        T02 = dst_mean[0] - scale * (T00*src_mean[0] + T01*src_mean[1])
        T12 = dst_mean[1] - scale * (T10*src_mean[0] + T11*src_mean[1])
        
        return [T00*scale, T01*scale, T02, T10*scale, T11*scale, T12]


class FaceRecognitionModule:
    """人脸识别模块"""
    
    def __init__(self, uart, protocol, rgb888p_size, display_size, 
                 database_dir, face_threshold=0.65, debug_mode=0):
        self.uart = uart
        self.protocol = protocol
        self.rgb888p_size = rgb888p_size
        self.display_size = display_size
        self.database_dir = database_dir
        self.face_threshold = face_threshold
        self.debug_mode = debug_mode
        
        self.face_det = None
        self.face_reg = None
        self.anchors = None
        
        # 人脸数据库
        self.db_names = []
        self.db_features = []
        
        self.initialized = False
        self.running = False
    
    def init(self):
        """初始化（加载模型）"""
        if self.initialized:
            return
        
        print("[FaceRec] Loading models...")
        
        # 加载 anchors
        self.anchors = np.fromfile(ANCHORS_PATH, dtype=np.float)
        self.anchors = self.anchors.reshape((4200, 4))
        
        # 创建检测器
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
        
        # 创建特征提取器
        self.face_reg = FaceFeatureApp(
            kmodel_path=FACE_REG_KMODEL,
            model_input_size=FACE_REG_INPUT_SIZE,
            rgb888p_size=self.rgb888p_size,
            debug_mode=self.debug_mode
        )
        
        # 加载数据库
        self._load_database()
        
        self.initialized = True
        print("[FaceRec] Models loaded, %d faces in database" % len(self.db_names))
    
    def _load_database(self):
        """加载人脸数据库"""
        self.db_names = []
        self.db_features = []
        
        try:
            for item in os.listdir(self.database_dir):
                if item.endswith('.bin'):
                    filepath = self.database_dir + item
                    with open(filepath, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_features.append(feature)
                    self.db_names.append(item[:-4])
        except Exception as e:
            print("[FaceRec] Load database error:", e)
    
    def reload_database(self):
        """重新加载数据库"""
        self._load_database()
        print("[FaceRec] Database reloaded, %d faces" % len(self.db_names))
    
    def start(self):
        self.running = True
        print("[FaceRec] Started")
    
    def stop(self):
        self.running = False
        print("[FaceRec] Stopped")
    
    def is_running(self):
        return self.running and self.initialized
    
    def run_once(self, pl):
        """执行一次识别"""
        if not self.is_running():
            return None, None
        
        with ScopedTiming("face_rec_total", self.debug_mode > 0):
            # 获取帧
            img = pl.get_frame()
            
            # 人脸检测
            det_boxes, landms = self.face_det.run(img)
            
            # 识别每个人脸（限制数量避免资源耗尽）
            recg_results = []
            if det_boxes is not None and landms is not None:
                # 【修复】每帧最多处理2个人脸，避免资源耗尽
                max_faces = min(len(landms), 2)
                
                for i in range(max_faces):
                    landm = landms[i]
                    # 配置预处理并提取特征
                    self.face_reg.config_preprocess(landm)
                    feature = self.face_reg.run(img)
                    name, score = self._match_feature(feature)
                    recg_results.append((name, score))
                
                # 剩余人脸标记为 unknown
                for i in range(max_faces, len(det_boxes)):
                    recg_results.append(("unknown", 0.0))
            
            # 绘制和发送
            self._draw_and_send(pl, det_boxes, recg_results)
            
            # 显示
            pl.show_image()
            
            # 【修复】主动垃圾回收
            gc.collect()
        
        return det_boxes, recg_results
    
    def _match_feature(self, feature):
        """匹配特征"""
        if len(self.db_names) == 0:
            return "unknown", 0.0
        
        # 归一化
        norm = np.linalg.norm(feature)
        if norm > 0:
            feature = feature / norm
        
        best_name = "unknown"
        best_score = 0.0
        
        for i in range(len(self.db_names)):
            db_feat = self.db_features[i]
            db_norm = np.linalg.norm(db_feat)
            if db_norm > 0:
                db_feat = db_feat / db_norm
            
            score = float(np.dot(feature, db_feat) / 2 + 0.5)
            
            if score > best_score:
                best_score = score
                if score >= self.face_threshold:
                    best_name = self.db_names[i]
        
        return best_name, best_score
    
    def _draw_and_send(self, pl, det_boxes, recg_results):
        """绘制结果并发送"""
        pl.osd_img.clear()
        
        if det_boxes is None:
            return
        
        for i, det in enumerate(det_boxes):
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            
            name = "unknown"
            score = 0.0
            if i < len(recg_results):
                name, score = recg_results[i]
            
            # 坐标转换
            x_d = x * self.display_size[0] // self.rgb888p_size[0]
            y_d = y * self.display_size[1] // self.rgb888p_size[1]
            w_d = w * self.display_size[0] // self.rgb888p_size[0]
            h_d = h * self.display_size[1] // self.rgb888p_size[1]
            
            # 颜色
            color = (255, 0, 255, 0) if name != "unknown" else (255, 255, 0, 0)
            
            # 绘制
            pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, color=color, thickness=2)
            text = "%s:%.0f%%" % (name, score * 100)
            pl.osd_img.draw_string_advanced(x_d, y_d - 25, 20, text, color=(255, 255, 255, 0))
            
            # 发送
            data = self.protocol.get_face_recognition_data(x, y, w, h, name, score)
            self.uart.send(data)
    
    def deinit(self):
        """释放资源"""
        print("[FaceRec] Deinitializing...")
        
        # 【修复】先释放 face_reg 的 ai2d
        if self.face_reg:
            if hasattr(self.face_reg, 'ai2d') and self.face_reg.ai2d is not None:
                try:
                    del self.face_reg.ai2d
                    self.face_reg.ai2d = None
                except:
                    pass
            self.face_reg.deinit()
            self.face_reg = None
        
        if self.face_det:
            self.face_det.deinit()
            self.face_det = None
        
        self.anchors = None
        self.db_names = []
        self.db_features = []
        self.initialized = False
        self.running = False
        
        gc.collect()
        print("[FaceRec] Deinitialized")