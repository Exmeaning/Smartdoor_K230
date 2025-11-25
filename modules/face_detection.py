"""
人脸检测模块 - 延迟加载版
"""

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import gc

# 配置
FACE_DET_KMODEL = "/sdcard/kmodel/face_detection_320.kmodel"
ANCHORS_PATH = "/sdcard/utils/prior_data_320.bin"
FACE_DET_INPUT_SIZE = [320, 320]
ANCHOR_LEN = 4200
DET_DIM = 4
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.2


class FaceDetectionApp(AIBase):
    """人脸检测 AI 应用"""
    
    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold, nms_threshold, 
                 rgb888p_size, display_size, debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        self.model_input_size = model_input_size
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.rgb888p_size = rgb888p_size
        self.display_size = display_size
        
        # 创建预处理器
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT,
                                  np.uint8, np.uint8)
    
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("det preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            top, bottom, left, right = self._get_padding()
            
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
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
                self.rgb888p_size,
                results
            )
            if len(res) == 0:
                return None, None
            return res[0], res[1]
    
    def _get_padding(self):
        dst_w, dst_h = self.model_input_size
        src_w, src_h = self.rgb888p_size
        ratio = min(dst_w / src_w, dst_h / src_h)
        new_w = int(ratio * src_w)
        new_h = int(ratio * src_h)
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        return (0, int(round(dh * 2 + 0.1)), 0, int(round(dw * 2 - 0.1)))


class FaceDetectionModule:
    """人脸检测模块"""
    
    def __init__(self, uart, protocol, rgb888p_size, display_size, debug_mode=0):
        self.uart = uart
        self.protocol = protocol
        self.rgb888p_size = rgb888p_size
        self.display_size = display_size
        self.debug_mode = debug_mode
        
        self.face_det = None
        self.anchors = None
        
        self.initialized = False
        self.running = False
    
    def init(self):
        """初始化（加载模型）"""
        if self.initialized:
            return
        
        print("[FaceDet] Loading model...")
        
        # 加载 anchors
        self.anchors = np.fromfile(ANCHORS_PATH, dtype=np.float)
        self.anchors = self.anchors.reshape((ANCHOR_LEN, DET_DIM))
        
        # 创建检测器
        self.face_det = FaceDetectionApp(
            kmodel_path=FACE_DET_KMODEL,
            model_input_size=FACE_DET_INPUT_SIZE,
            anchors=self.anchors,
            confidence_threshold=CONFIDENCE_THRESHOLD,
            nms_threshold=NMS_THRESHOLD,
            rgb888p_size=self.rgb888p_size,
            display_size=self.display_size,
            debug_mode=self.debug_mode
        )
        
        # 配置预处理
        self.face_det.config_preprocess()
        
        self.initialized = True
        print("[FaceDet] Model loaded")
    
    def start(self):
        self.running = True
        print("[FaceDet] Started")
    
    def stop(self):
        self.running = False
        print("[FaceDet] Stopped")
    
    def is_running(self):
        return self.running and self.initialized
    
    def run_once(self, pl):
        """执行一次检测"""
        if not self.is_running():
            return None
        
        with ScopedTiming("face_det_total", self.debug_mode > 0):
            # 获取帧
            img = pl.get_frame()
            
            # 运行检测
            det_boxes, landms = self.face_det.run(img)
            
            # 绘制和发送
            self._draw_and_send(pl, det_boxes)
            
            # 显示
            pl.show_image()
        
        return det_boxes
    
    def _draw_and_send(self, pl, det_boxes):
        """绘制结果并发送数据"""
        pl.osd_img.clear()
        
        if det_boxes is None:
            return
        
        for det in det_boxes:
            x, y, w, h = int(det[0]), int(det[1]), int(det[2]), int(det[3])
            
            # 坐标转换
            x_d = x * self.display_size[0] // self.rgb888p_size[0]
            y_d = y * self.display_size[1] // self.rgb888p_size[1]
            w_d = w * self.display_size[0] // self.rgb888p_size[0]
            h_d = h * self.display_size[1] // self.rgb888p_size[1]
            
            # 绘制
            pl.osd_img.draw_rectangle(x_d, y_d, w_d, h_d, 
                                       color=(255, 255, 0, 255), thickness=2)
            
            # 发送
            data = self.protocol.get_face_detect_data(x, y, w, h)
            self.uart.send(data)
    
    def deinit(self):
        """释放资源"""
        print("[FaceDet] Deinitializing...")
        
        if self.face_det:
            self.face_det.deinit()
            self.face_det = None
        
        self.anchors = None
        self.initialized = False
        self.running = False
        
        gc.collect()
        print("[FaceDet] Deinitialized")