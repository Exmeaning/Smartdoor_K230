"""
人脸检测功能模块 - 单帧处理模式
"""

from libs.PipeLine import PipeLine, ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import gc

class FaceDetectionApp(AIBase):
    """人脸检测应用类"""
    
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
            top, bottom, left, right = self.get_padding_param()
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            self.ai2d.build([1, 3, ai2d_input_size[1], ai2d_input_size[0]],
                           [1, 3, self.model_input_size[1], self.model_input_size[0]])
    
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            post_ret = aidemo.face_det_post_process(
                self.confidence_threshold,
                self.nms_threshold,
                self.model_input_size[1],
                self.anchors,
                self.rgb888p_size,
                results
            )
            return post_ret[0] if post_ret else post_ret
    
    def get_padding_param(self):
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)
        
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        
        return (int(round(0)), int(round(dh * 2 + 0.1)),
                int(round(0)), int(round(dw * 2 - 0.1)))


def face_detect_init(controller):
    """初始化人脸检测"""
    print("[FaceDetect] Initializing...")
    
    kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((4200, 4))
    
    conf_threshold = controller.config.get('detect_threshold', 0.5)
    nms_threshold = controller.config.get('nms_threshold', 0.2)
    
    rgb888p_size = [640, 480]
    display_size = [640, 480]
    
    # 创建Pipeline
    pl = PipeLine(rgb888p_size=rgb888p_size, display_size=display_size, display_mode="lcd")
    pl.create()
    
    # 创建检测器
    face_det = FaceDetectionApp(
        kmodel_path,
        model_input_size=[320, 320],
        anchors=anchors,
        confidence_threshold=conf_threshold,
        nms_threshold=nms_threshold,
        rgb888p_size=rgb888p_size,
        display_size=display_size,
        debug_mode=0
    )
    face_det.config_preprocess()
    
    print("[FaceDetect] Initialized")
    
    return {
        'pipeline': pl, 
        'detector': face_det, 
        'display_size': display_size, 
        'rgb888p_size': rgb888p_size,
        'frame_count': 0
    }


def face_detect_handler(controller, stop_check):
    """
    人脸检测处理函数 - 每次调用处理一帧
    注意：这个函数会被主循环反复调用，不是自己循环！
    """
    obj = controller.current_func_obj
    if obj is None:
        return
    
    pl = obj['pipeline']
    face_det = obj['detector']
    display_size = obj['display_size']
    rgb888p_size = obj['rgb888p_size']
    
    try:
        # 获取帧
        img = pl.get_frame()
        
        # 运行检测
        dets = face_det.run(img)
        
        # 清除OSD
        pl.osd_img.clear()
        
        # 处理结果
        if dets:
            for det in dets:
                x, y, w, h = map(lambda v: int(round(v, 0)), det[:4])
                
                # 坐标转换
                x_disp = x * display_size[0] // rgb888p_size[0]
                y_disp = y * display_size[1] // rgb888p_size[1]
                w_disp = w * display_size[0] // rgb888p_size[0]
                h_disp = h * display_size[1] // rgb888p_size[1]
                
                # 绘制
                pl.osd_img.draw_rectangle(x_disp, y_disp, w_disp, h_disp, 
                                          color=(255, 255, 0, 255), thickness=2)
                
                # 发送数据
                controller.send_face_detect(x, y, w, h)
        
        # 显示
        pl.show_image()
        
        # 定期GC
        obj['frame_count'] = obj.get('frame_count', 0) + 1
        if obj['frame_count'] % 30 == 0:
            gc.collect()
        
    except Exception as e:
        print("[FaceDetect] Frame error:", e)


def face_detect_deinit(obj):
    """清理人脸检测"""
    print("[FaceDetect] Deinitializing...")
    try:
        if obj and 'detector' in obj:
            obj['detector'].deinit()
        if obj and 'pipeline' in obj:
            obj['pipeline'].destroy()
    except Exception as e:
        print("[FaceDetect] Deinit error:", e)
    print("[FaceDetect] Deinitialized")