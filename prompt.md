# K230相关生成提示词

## 现在有一个K230开发板 基于mircoPython 还有一个 树莓派 需要使用主从架构 生成一个树莓派控制K230的方案 server还有一个go后端（这个已经写好了 到时候写树莓派端才想你提问）是一个采用webscoket的远程通信
## 你有最大输出限制 请尽量分多次输出以免超出限制

- micropython许多东西具有限制 我把对应的API手册告诉你
- 例如串口部分的API文档 等等

以下是一些例程代码： K230端
```PYTHON
from libs.PipeLine import PipeLine, ScopedTiming
# Pipeline是集成了摄像头、LCD屏幕显示等多媒体功能的模块，大大简化了调用摄像头和显示相关的操作
# Pipeline is a module that integrates multimedia functions like camera and LCD display, greatly simplifying camera and display operations
​
# ScopedTiming是一个代码执行计时器，使用方法可以参考例程代码
# ScopedTiming is a code execution timer, refer to example code for usage
​
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
# AIBase和AI2D用来处理AI底层的逻辑
# AIBase is the base class for all AI functionality implementations
# AI2D用于处理图像
# AI2D is used for image processing
​
import os
import ujson
# os和ujson分别提供系统相关操作和JSON数据相关操作，不是每个例程都能用上
# os and ujson provide system operations and JSON data operations respectively, not required for every example
​
from media.media import *
from time import *
import nncase_runtime as nn
# nncase是K230进行AI推理的核心模块，提供了便捷的方法供用户调用K230的KPU
# nncase is the core module for AI inference on K230, providing convenient methods for users to call K230's KPU
​
import ulab.numpy as np
# ulab.numpy是从python的numpy中移植而来，用于进行一些AI运算中必要的矩阵操作
# ulab.numpy is ported from Python's numpy, used for necessary matrix operations in AI computations
​
import time
import utime
import image
import random
import gc
import sys
import aidemo
# aidemo也是核心模块之一，K230固件中预制了非常多的AI玩法
# aidemo is another core module, K230 firmware includes many pre-configured AI applications
# 通过aidemo模块可以快速简单的调用这些玩法中的复杂方法
# Complex methods in these applications can be easily called through the aidemo module
​
import _thread
# _thread是线程模块，前面章节中我们有做过详细讲解，这里就不再赘述
# _thread is the threading module, as detailed in previous chapters, no need to elaborate further
 

自定义人脸检测类
# 自定义人脸检测类，继承自AIBase基类
class FaceDetectionApp(AIBase):
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.5, nms_threshold=0.2, rgb888p_size=[224,224], display_size=[1920,1080], debug_mode=0):
        # 调用基类的构造函数 / Call parent class constructor
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)
        
        # 模型文件路径 / Path to the model file
        self.kmodel_path = kmodel_path
        
        # 模型输入分辨率 / Model input resolution
        self.model_input_size = model_input_size
        
        # 置信度阈值：检测结果的最小置信度要求 / Confidence threshold: minimum confidence requirement for detection results
        self.confidence_threshold = confidence_threshold
        
        # NMS阈值：非极大值抑制的阈值 / NMS threshold: threshold for Non-Maximum Suppression
        self.nms_threshold = nms_threshold
        
        # 锚点数据：用于目标检测的预定义框 / Anchor data: predefined boxes for object detection
        self.anchors = anchors
        
        # sensor给到AI的图像分辨率，宽度16对齐 / Image resolution from sensor to AI, width aligned to 16
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0], 16), rgb888p_size[1]]
        
        # 显示分辨率，宽度16对齐 / Display resolution, width aligned to 16
        self.display_size = [ALIGN_UP(display_size[0], 16), display_size[1]]
        
        # 调试模式标志 / Debug mode flag
        self.debug_mode = debug_mode
        
        # 实例化AI2D对象用于图像预处理 / Initialize AI2D object for image preprocessing
        self.ai2d = Ai2d(debug_mode)
        
        # 设置AI2D的输入输出格式 / Set AI2D input/output format
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT, nn.ai2d_format.NCHW_FMT, np.uint8, np.uint8)
​
    def config_preprocess(self, input_image_size=None):
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 获取AI2D输入尺寸 / Get AI2D input size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            
            # 获取padding参数 / Get padding parameters
            top, bottom, left, right = self.get_padding_param()
            
            # 设置padding: [上,下,左,右], 填充值[104,117,123] / Set padding: [top,bottom,left,right], padding value[104,117,123]
            self.ai2d.pad([0, 0, 0, 0, top, bottom, left, right], 0, [104, 117, 123])
            
            # 设置resize方法：双线性插值 / Set resize method: bilinear interpolation
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            
            # 构建预处理流程 / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])
​
    def postprocess(self, results):
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 调用aidemo库进行人脸检测后处理 / Call aidemo library for face detection post-processing
            post_ret = aidemo.face_det_post_process(self.confidence_threshold, 
                                                  self.nms_threshold,
                                                  self.model_input_size[1], 
                                                  self.anchors,
                                                  self.rgb888p_size,
                                                  results)
            return post_ret[0] if post_ret else post_ret
​
    def draw_result(self, pl, dets):
        with ScopedTiming("display_draw", self.debug_mode > 0):
            if dets:
                # 清除上一帧的OSD绘制 / Clear previous frame's OSD drawing
                pl.osd_img.clear()
                
                for det in dets:
                    # 转换检测框坐标到显示分辨率 / Convert detection box coordinates to display resolution
                    x, y, w, h = map(lambda x: int(round(x, 0)), det[:4])
                    x = x * self.display_size[0] // self.rgb888p_size[0]
                    y = y * self.display_size[1] // self.rgb888p_size[1]
                    w = w * self.display_size[0] // self.rgb888p_size[0]
                    h = h * self.display_size[1] // self.rgb888p_size[1]
                    
                    # 绘制黄色检测框 / Draw yellow detection box
                    pl.osd_img.draw_rectangle(x, y, w, h, color=(255, 255, 0, 255), thickness=2)
            else:
                pl.osd_img.clear()
​
    def get_padding_param(self):
        # 计算模型输入和实际图像的缩放比例 / Calculate scaling ratio between model input and actual image
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)
        
        # 计算缩放后的新尺寸 / Calculate new dimensions after scaling
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])
        
        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2
        
        # 返回padding参数 / Return padding parameters
        return (int(round(0)),
                int(round(dh * 2 + 0.1)),
                int(round(0)),
                int(round(dw * 2 - 0.1)))
​
 

执行人脸检测
def exce_demo(pl):
    # 声明全局变量face_det / Declare global variable face_det
    global face_det
​
    # 获取显示相关参数 / Get display-related parameters
    display_mode = pl.display_mode      # 显示模式(如lcd) / Display mode (e.g., lcd)
    rgb888p_size = pl.rgb888p_size     # 原始图像分辨率 / Original image resolution
    display_size = pl.display_size      # 显示分辨率 / Display resolution
​
    # 设置人脸检测模型路径 / Set face detection model path
    kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
​
    # 设置模型参数 / Set model parameters
    confidence_threshold = 0.5    # 置信度阈值 / Confidence threshold
    nms_threshold = 0.2          # 非极大值抑制阈值 / Non-maximum suppression threshold
    anchor_len = 4200            # 锚框数量 / Number of anchor boxes
    det_dim = 4                  # 检测维度(x,y,w,h) / Detection dimensions (x,y,w,h)
    
    # 加载锚框数据 / Load anchor box data
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))
​
    try:
        # 初始化人脸检测应用实例 / Initialize face detection application instance
        face_det = FaceDetectionApp(kmodel_path, 
                                  model_input_size=[320, 320], 
                                  anchors=anchors,
                                  confidence_threshold=confidence_threshold,
                                  nms_threshold=nms_threshold,
                                  rgb888p_size=rgb888p_size,
                                  display_size=display_size,
                                  debug_mode=0)
        
        # 配置图像预处理 / Configure image preprocessing
        face_det.config_preprocess()
​
        # 主循环 / Main loop
        while True:
            with ScopedTiming("total",0):    # 计时器 / Timer
                img = pl.get_frame()          # 获取摄像头帧图像 / Get camera frame
                res = face_det.run(img)       # 执行人脸检测 / Run face detection
                face_det.draw_result(pl, res) # 绘制检测结果 / Draw detection results
                pl.show_image()               # 显示处理后的图像 / Display processed image
                gc.collect()                  # 垃圾回收 / Garbage collection
                time.sleep_us(10)             # 短暂延时 / Brief delay
​
    except Exception as e:
        print("人脸检测功能退出")           # 异常退出提示 / Exception exit prompt
    finally:
        face_det.deinit()                   # 释放资源 / Release resources

```
人脸注册
```PYTHON
# 导入必要的库文件 / Import necessary libraries
from libs.PipeLine import PipeLine, ScopedTiming  # 导入Pipeline和计时工具 / Import pipeline and timing tools
from libs.AIBase import AIBase     # 导入AI基础类 / Import AI base class
from libs.AI2D import Ai2d        # 导入AI 2D处理类 / Import AI 2D processing class
import os                         # 导入操作系统接口 / Import OS interface
import ujson                      # 导入JSON处理库 / Import JSON processing library
from media.media import *         # 导入媒体处理库 / Import media processing library
from time import *               # 导入时间处理库 / Import time processing library
import nncase_runtime as nn      # 导入神经网络运行时 / Import neural network runtime
import ulab.numpy as np          # 导入numpy库 / Import numpy library
import time                      # 导入时间库 / Import time library
import image                     # 导入图像处理库 / Import image processing library
import aidemo                    # 导入AI演示库 / Import AI demo library
import random                    # 导入随机数库 / Import random number library
import gc                        # 导入垃圾回收库 / Import garbage collection library
import sys                       # 导入系统库 / Import system library
import math                      # 导入数学库 / Import math library

global fr                        # 声明全局变量 / Declare global variable

class FaceDetApp(AIBase):
    """人脸检测应用类 / Face Detection Application Class

    这个类继承自AIBase，实现了人脸检测的功能
    This class inherits from AIBase and implements face detection functionality
    """

    def __init__(self, kmodel_path, model_input_size, anchors,
                 confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1280,720], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function

        参数 / Parameters:
        - kmodel_path: KPU模型的路径 / Path to KPU model
        - model_input_size: 模型输入尺寸 / Model input size
        - anchors: 锚框参数 / Anchor box parameters
        - confidence_threshold: 置信度阈值 / Confidence threshold
        - nms_threshold: NMS阈值 / NMS threshold
        - rgb888p_size: RGB888格式图像尺寸 / RGB888 format image size
        - display_size: 显示尺寸 / Display size
        - debug_mode: 调试模式 / Debug mode
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path                # KPU模型路径 / KPU model path
        self.model_input_size = model_input_size      # 模型输入尺寸 / Model input size
        self.confidence_threshold = confidence_threshold  # 置信度阈值 / Confidence threshold
        self.nms_threshold = nms_threshold            # NMS阈值 / NMS threshold
        self.anchors = anchors                        # 锚框参数 / Anchor box parameters

        # 设置RGB888图像尺寸，确保宽度16字节对齐 / Set RGB888 image size, ensure width is 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]

        # 设置显示尺寸，确保宽度16字节对齐 / Set display size, ensure width is 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]

        self.debug_mode = debug_mode                  # 调试模式 / Debug mode

        # 初始化AI2D对象，用于图像预处理 / Initialize AI2D object for image preprocessing
        self.ai2d = Ai2d(debug_mode)

        # 设置AI2D的数据类型和格式 / Set AI2D data type and format
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)
        self.image_size = []

    def config_preprocess(self, input_image_size=None):
        """配置预处理参数 / Configure preprocessing parameters

        对输入图像进行pad和resize等预处理操作
        Perform preprocessing operations such as pad and resize on input images
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 设置输入图像尺寸 / Set input image size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size
            self.image_size = [input_image_size[1], input_image_size[0]]

            # 配置padding参数 / Configure padding parameters
            self.ai2d.pad(self.get_pad_param(ai2d_input_size), 0, [104,117,123])

            # 配置resize参数 / Configure resize parameters
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)

            # 构建预处理Pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                           [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法 / Post-processing method

        处理模型的原始输出，得到最终的检测结果
        Process the model's raw output to get final detection results
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 调用aidemo库进行人脸检测后处理 / Call aidemo library for face detection post-processing
            res = aidemo.face_det_post_process(self.confidence_threshold,
                                             self.nms_threshold,
                                             self.model_input_size[0],
                                             self.anchors,
                                             self.image_size,
                                             results)
            if len(res) == 0:
                return res
            else:
                return res[0], res[1]

    def get_pad_param(self, image_input_size):
        """计算padding参数 / Calculate padding parameters

        计算等比例缩放后需要的padding参数
        Calculate the padding parameters needed after proportional scaling
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]

        # 计算缩放比例 / Calculate scaling ratio
        ratio_w = dst_w / image_input_size[0]
        ratio_h = dst_h / image_input_size[1]
        ratio = min(ratio_w, ratio_h)

        # 计算新的尺寸 / Calculate new dimensions
        new_w = int(ratio * image_input_size[0])
        new_h = int(ratio * image_input_size[1])

        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2

        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))

        return [0,0,0,0,top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    """人脸注册应用类 / Face Registration Application Class

    处理人脸注册相关的功能
    Handle face registration related functions
    """

    def __init__(self, kmodel_path, model_input_size,
                 rgb888p_size=[1920,1080], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function"""
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        self.kmodel_path = kmodel_path                # 模型路径 / Model path
        self.model_input_size = model_input_size      # 模型输入尺寸 / Model input size
        # RGB尺寸，确保16字节对齐 / RGB size, ensure 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        # 显示尺寸，确保16字节对齐 / Display size, ensure 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode                  # 调试模式 / Debug mode

        # 标准5个关键点坐标 / Standard 5 keypoint coordinates
        self.umeyama_args_112 = [
            38.2946 , 51.6963,
            73.5318 , 51.5014,
            56.0252 , 71.7366,
            41.5493 , 92.3655,
            70.7299 , 92.2041
        ]

        # 初始化AI2D / Initialize AI2D
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        """配置预处理参数 / Configure preprocessing parameters"""
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 计算仿射变换矩阵并配置 / Calculate and configure affine transformation matrix
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)

            # 构建预处理Pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                           [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """后处理方法 / Post-processing method"""
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        """2x2矩阵的奇异值分解 / Singular Value Decomposition for 2x2 matrix"""
        s = [0.0, 0.0]
        u = [0.0, 0.0, 0.0, 0.0]
        v = [0.0, 0.0, 0.0, 0.0]

        # 计算奇异值 / Calculate singular values
        s[0] = (math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2) +
                math.sqrt((a[0] + a[3]) ** 2 + (a[1] - a[2]) ** 2)) / 2
        s[1] = abs(s[0] - math.sqrt((a[0] - a[3]) ** 2 + (a[1] + a[2]) ** 2))

        # 计算右奇异向量 / Calculate right singular vectors
        v[2] = math.sin((math.atan2(2 * (a[0] * a[1] + a[2] * a[3]),
                                   a[0] ** 2 - a[1] ** 2 + a[2] ** 2 - a[3] ** 2)) / 2) if s[0] > s[1] else 0
        v[0] = math.sqrt(1 - v[2] ** 2)
        v[1] = -v[2]
        v[3] = v[0]

        # 计算左奇异向量 / Calculate left singular vectors
        u[0] = -(a[0] * v[0] + a[1] * v[2]) / s[0] if s[0] != 0 else 1
        u[2] = -(a[2] * v[0] + a[3] * v[2]) / s[0] if s[0] != 0 else 0
        u[1] = (a[0] * v[1] + a[1] * v[3]) / s[1] if s[1] != 0 else -u[2]
        u[3] = (a[2] * v[1] + a[3] * v[3]) / s[1] if s[1] != 0 else u[0]

        v[0] = -v[0]
        v[2] = -v[2]

        return u, s, v

    def image_umeyama_112(self, src):
        """使用Umeyama算法计算仿射变换矩阵 / Calculate affine transformation matrix using Umeyama algorithm"""
        SRC_NUM = 5
        SRC_DIM = 2

        # 计算源点和目标点的均值 / Calculate mean of source and target points
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

        # 去中心化 / De-mean
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]

        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        # 计算协方差矩阵 / Calculate covariance matrix
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        # SVD分解和旋转矩阵计算 / SVD decomposition and rotation matrix calculation
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])

        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        # 计算缩放因子 / Calculate scaling factor
        scale = 1.0
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

        # 计算平移向量 / Calculate translation vector
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])

        # 应用缩放 / Apply scaling
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale

        return T

    def get_affine_matrix(self, sparse_points):
        """获取仿射变换矩阵 / Get affine transformation matrix"""
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                         matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst

class FaceRegistration:
    """人脸注册主类 / Main Face Registration Class

    整合人脸检测和注册功能的主类
    Main class that integrates face detection and registration functions
    """

    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size,
                 reg_input_size, database_dir, anchors,
                 confidence_threshold=0.25, nms_threshold=0.3,
                 rgb888p_size=[1280,720], display_size=[1920,1080],
                 debug_mode=0):
        """初始化函数 / Initialization function"""
        # 人脸检测模型路径 / Face detection model path
        self.face_det_kmodel = face_det_kmodel
        # 人脸注册模型路径 / Face registration model path
        self.face_reg_kmodel = face_reg_kmodel
        # 人脸检测模型输入尺寸 / Face detection model input size
        self.det_input_size = det_input_size
        # 人脸注册模型输入尺寸 / Face registration model input size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        # RGB尺寸，确保16字节对齐 / RGB size, ensure 16-byte aligned
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        # 显示尺寸，确保16字节对齐 / Display size, ensure 16-byte aligned
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 初始化人脸检测和注册模型 / Initialize face detection and registration models
        self.face_det = FaceDetApp(self.face_det_kmodel,
                                 model_input_size=self.det_input_size,
                                 anchors=self.anchors,
                                 confidence_threshold=self.confidence_threshold,
                                 nms_threshold=self.nms_threshold,
                                 debug_mode=0)
        self.face_reg = FaceRegistrationApp(self.face_reg_kmodel,
                                          model_input_size=self.reg_input_size,
                                          rgb888p_size=self.rgb888p_size)

    def run(self, input_np, img_file):
        """运行人脸注册流程 / Run face registration process"""
        # 配置人脸检测预处理 / Configure face detection preprocessing
        self.face_det.config_preprocess(input_image_size=[input_np.shape[3],input_np.shape[2]])
        # 执行人脸检测 / Perform face detection
        det_boxes, landms = self.face_det.run(input_np)

        try:
            if det_boxes:
                if det_boxes.shape[0] == 1:
                    # 若只检测到一张人脸，进行注册 / If only one face is detected, proceed with registration
                    db_i_name = img_file.split('.')[0]
                    for landm in landms:
                        # 配置人脸注册预处理 / Configure face registration preprocessing
                        self.face_reg.config_preprocess(landm, input_image_size=[input_np.shape[3],input_np.shape[2]])
                        # 执行人脸特征提取 / Perform face feature extraction
                        reg_result = self.face_reg.run(input_np)
                        # 保存特征到数据库 / Save features to database
                        with open(self.database_dir+'{}.bin'.format(db_i_name), "wb") as file:
                            file.write(reg_result.tobytes())
                            print('Success!')
                else:
                    print('Only one person in a picture when you sign up')
            else:
                print('No person detected')
        except:
            print("Register failed")

    def image2rgb888array(self, img):
        """将图像转换为RGB888数组 / Convert image to RGB888 array"""
        with ScopedTiming("fr_kpu_deinit", self.debug_mode > 0):
            # 转换为RGB888格式 / Convert to RGB888 format
            img_data_rgb888 = img.to_rgb888()
            # 转换为numpy数组 / Convert to numpy array
            img_hwc = img_data_rgb888.to_numpy_ref()
            shape = img_hwc.shape
            # 重塑并转置数组 / Reshape and transpose array
            img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
            img_tmp_trans = img_tmp.transpose()
            img_res = img_tmp_trans.copy()
            # 返回NCHW格式的数组 / Return array in NCHW format
            img_return = img_res.reshape((1, shape[2], shape[0], shape[1]))
        return img_return

def ensure_dir(directory):
    """
    递归创建目录
    (Recursively create directory)
    """
    # 如果目录为空字符串或根目录，直接返回
    # (If directory is empty string or root directory, return directly)
    if not directory or directory == '/':
        return

    # 处理路径分隔符，确保使用标准格式
    # (Process path separators to ensure standard format)
    directory = directory.rstrip('/')

    try:
        # 尝试获取目录状态，如果目录存在就直接返回
        # (Try to get directory status, if directory exists then return directly)
        os.stat(directory)
        print(f'目录已存在: {directory}')
        # (Directory already exists: {directory})
        return
    except OSError:
        # 目录不存在，需要创建
        # (Directory does not exist, need to create)

        # 分割路径以获取父目录
        # (Split path to get parent directory)
        if '/' in directory:
            parent = directory[:directory.rindex('/')]
            if parent and parent != directory:  # 避免无限递归
                                                # (Avoid infinite recursion)
                ensure_dir(parent)

        try:
            # 创建目录
            # (Create directory)
            os.mkdir(directory)
            print(f'已创建目录: {directory}')
            # (Directory created: {directory})
        except OSError as e:
            # 可能是并发创建导致的冲突，再次检查目录是否存在
            # (Possible conflict due to concurrent creation, check again if directory exists)
            try:
                os.stat(directory)
                print(f'目录已被其他进程创建: {directory}')
                # (Directory has been created by another process: {directory})
            except:
                # 如果仍然不存在，则确实出错了
                # (If it still doesn't exist, there is definitely an error)
                print(f'创建目录时出错: {e}')
                # (Error creating directory: {e})
    except Exception as e:
        # 捕获其他可能的异常
        # (Catch other possible exceptions)
        print(f'处理目录时出错: {e}')
        # (Error processing directory: {e})

def get_directory_name(path):
    """获取路径中的目录名 / Get directory name from path"""
    parts = path.split('/')
    for part in reversed(parts):
        if part:
            return part
    return ''

def exce_demo(pl=None):
    """执行演示的主函数 / Main function to execute demonstration"""
    global eg

    # 配置模型和参数路径 / Configure model and parameter paths
    face_det_kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"

    # 此处需要修改为你的人脸照片所在的目录
    # change this path to where your face photo in
    database_img_dir = "/data/photo/931783/"
    dir_name = get_directory_name(database_img_dir)
    face_det_input_size = [320,320]
    face_reg_input_size = [112,112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4

    # 加载anchors数据 / Load anchors data
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))

    # 设置最大注册人脸数和特征维度 / Set maximum number of registered faces and feature dimensions
    max_register_face = 100
    feature_num = 128

    print("Start ...")
    database_dir = "/data/face_database/" + dir_name + "/"
    ensure_dir(database_dir)

    # 初始化人脸注册对象 / Initialize face registration object
    fr = FaceRegistration(face_det_kmodel_path, face_reg_kmodel_path,
                         det_input_size=face_det_input_size,
                         reg_input_size=face_reg_input_size,
                         database_dir=database_dir,
                         anchors=anchors,
                         confidence_threshold=confidence_threshold,
                         nms_threshold=nms_threshold)

    # 获取图像列表并处理 / Get image list and process
    img_list = os.listdir(database_img_dir)
    try:
        for img_file in img_list:
            # 读取图像 / Read image
            full_img_file = database_img_dir + img_file
            print(full_img_file)
            img = image.Image(full_img_file)
            img.compress_for_ide()
            # 转换图像格式并处理 / Convert image format and process
            rgb888p_img_ndarry = fr.image2rgb888array(img)
            fr.run(rgb888p_img_ndarry, img_file)
            gc.collect()
    except Exception as e:
        print("人脸注册功能异常退出")
    finally:
        fr.face_det.deinit()
        fr.face_reg.deinit()
        print("人脸注册功能退出")

def exit_demo():
    """退出函数 / Exit function"""
    global fr
    # 清理资源 / Clean up resources
    fr.face_det.deinit()
    fr.face_reg.deinit()

if __name__ == "__main__":
    """程序入口 / Program entry"""
    exce_demo(None)
```
人脸识别
```PYTHON
# 导入所需库 / Import required libraries
from libs.PipeLine import PipeLine, ScopedTiming  # 导入视频处理Pipeline和计时器类 / Import video pipeline and timer classes
from libs.AIBase import AIBase                    # 导入AI基础类 / Import AI base class
from libs.AI2D import Ai2d                       # 导入AI 2D处理类 / Import AI 2D processing class
import os
import ujson
from media.media import *                        # 导入媒体处理相关库 / Import media processing libraries
from time import *
import nncase_runtime as nn                      # 导入神经网络运行时库 / Import neural network runtime library
import ulab.numpy as np                          # 导入类numpy库，用于数组操作 / Import numpy-like library for array operations
import time
import image                                     # 图像处理库 / Image processing library
import aidemo                                    # AI演示库 / AI demo library
import random
import gc                                        # 垃圾回收模块 / Garbage collection module
import sys
import math,re

# 全局变量定义 / Global variable definition
fr = None                                        # 人脸识别对象的全局变量 / Global variable for face recognition object
from libs.YbProtocol import YbProtocol
from ybUtils.YbUart import YbUart
# uart = None
uart = YbUart(baudrate=115200)
pto = YbProtocol()

class FaceDetApp(AIBase):
    """
    人脸检测应用类 / Face detection application class
    继承自AIBase基类 / Inherits from AIBase class
    """
    def __init__(self, kmodel_path, model_input_size, anchors, confidence_threshold=0.25,
                 nms_threshold=0.3, rgb888p_size=[640,480], display_size=[640,480], debug_mode=0):
        """
        初始化函数 / Initialization function
        参数说明 / Parameters:
        kmodel_path: 模型文件路径 / Model file path
        model_input_size: 模型输入尺寸 / Model input size
        anchors: 锚框参数 / Anchor box parameters
        confidence_threshold: 置信度阈值 / Confidence threshold
        nms_threshold: 非极大值抑制阈值 / Non-maximum suppression threshold
        rgb888p_size: 输入图像尺寸 / Input image size
        display_size: 显示尺寸 / Display size
        debug_mode: 调试模式标志 / Debug mode flag
        """
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        # 保存初始化参数 / Save initialization parameters
        self.kmodel_path = kmodel_path                  # kmodel文件路径 / kmodel file path
        self.model_input_size = model_input_size        # 模型输入尺寸 / Model input size
        self.confidence_threshold = confidence_threshold # 置信度阈值 / Confidence threshold
        self.nms_threshold = nms_threshold              # NMS阈值 / NMS threshold
        self.anchors = anchors                          # 锚框参数 / Anchor parameters

        # 图像尺寸处理（16字节对齐）/ Image size processing (16-byte alignment)
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]

        self.debug_mode = debug_mode                    # 调试模式 / Debug mode

        # 初始化AI2D预处理器 / Initialize AI2D preprocessor
        self.ai2d = Ai2d(debug_mode)
        # 设置AI2D参数 / Set AI2D parameters
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, input_image_size=None):
        """
        配置图像预处理参数 / Configure image preprocessing parameters
        使用pad和resize操作 / Use pad and resize operations
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            # 设置输入大小 / Set input size
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 配置padding参数 / Configure padding parameters
            self.ai2d.pad(self.get_pad_param(), 0, [104,117,123])
            # 配置resize参数 / Configure resize parameters
            self.ai2d.resize(nn.interp_method.tf_bilinear, nn.interp_mode.half_pixel)
            # 构建预处理pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """
        后处理方法 / Post-processing method
        使用aidemo库处理检测结果 / Process detection results using aidemo library
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            # 处理检测结果 / Process detection results
            res = aidemo.face_det_post_process(self.confidence_threshold,
                                             self.nms_threshold,
                                             self.model_input_size[0],
                                             self.anchors,
                                             self.rgb888p_size,
                                             results)
            # 返回检测结果 / Return detection results
            if len(res) == 0:
                return res, res
            else:
                return res[0], res[1]

    def get_pad_param(self):
        """
        计算padding参数 / Calculate padding parameters
        返回padding的边界值 / Return padding boundary values
        """
        dst_w = self.model_input_size[0]
        dst_h = self.model_input_size[1]

        # 计算缩放比例 / Calculate scaling ratio
        ratio_w = dst_w / self.rgb888p_size[0]
        ratio_h = dst_h / self.rgb888p_size[1]
        ratio = min(ratio_w, ratio_h)

        # 计算新的尺寸 / Calculate new dimensions
        new_w = int(ratio * self.rgb888p_size[0])
        new_h = int(ratio * self.rgb888p_size[1])

        # 计算padding值 / Calculate padding values
        dw = (dst_w - new_w) / 2
        dh = (dst_h - new_h) / 2

        # 返回padding参数 / Return padding parameters
        top = int(round(0))
        bottom = int(round(dh * 2 + 0.1))
        left = int(round(0))
        right = int(round(dw * 2 - 0.1))
        return [0, 0, 0, 0, top, bottom, left, right]

class FaceRegistrationApp(AIBase):
    """
    人脸注册应用类 / Face registration application class
    用于人脸特征提取和注册 / For face feature extraction and registration
    """
    def __init__(self, kmodel_path, model_input_size, rgb888p_size=[640,360],
                 display_size=[640,360], debug_mode=0):
        super().__init__(kmodel_path, model_input_size, rgb888p_size, debug_mode)

        # 初始化参数 / Initialize parameters
        self.kmodel_path = kmodel_path
        self.model_input_size = model_input_size
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 标准人脸关键点坐标 / Standard face keypoint coordinates
        self.umeyama_args_112 = [
            38.2946, 51.6963,
            73.5318, 51.5014,
            56.0252, 71.7366,
            41.5493, 92.3655,
            70.7299, 92.2041
        ]

        # 初始化AI2D / Initialize AI2D
        self.ai2d = Ai2d(debug_mode)
        self.ai2d.set_ai2d_dtype(nn.ai2d_format.NCHW_FMT,
                                nn.ai2d_format.NCHW_FMT,
                                np.uint8, np.uint8)

    def config_preprocess(self, landm, input_image_size=None):
        """
        配置预处理参数 / Configure preprocessing parameters
        使用仿射变换进行人脸对齐 / Use affine transformation for face alignment
        """
        with ScopedTiming("set preprocess config", self.debug_mode > 0):
            ai2d_input_size = input_image_size if input_image_size else self.rgb888p_size

            # 计算仿射变换矩阵 / Calculate affine transformation matrix
            affine_matrix = self.get_affine_matrix(landm)
            self.ai2d.affine(nn.interp_method.cv2_bilinear, 0, 0, 127, 1, affine_matrix)

            # 构建预处理pipeline / Build preprocessing pipeline
            self.ai2d.build([1,3,ai2d_input_size[1],ai2d_input_size[0]],
                          [1,3,self.model_input_size[1],self.model_input_size[0]])

    def postprocess(self, results):
        """
        后处理方法 / Post-processing method
        提取人脸特征 / Extract face features
        """
        with ScopedTiming("postprocess", self.debug_mode > 0):
            return results[0][0]

    def svd22(self, a):
        """
        2x2矩阵的奇异值分解 / Singular value decomposition for 2x2 matrix
        """
        # SVD计算 / SVD calculation
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
        """
        使用Umeyama算法进行人脸对齐 / Face alignment using Umeyama algorithm
        """
        SRC_NUM = 5  # 关键点数量 / Number of keypoints
        SRC_DIM = 2  # 坐标维度 / Coordinate dimensions

        # 计算源点和目标点的均值 / Calculate mean of source and target points
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

        # 去均值化 / De-mean
        src_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        dst_demean = [[0.0, 0.0] for _ in range(SRC_NUM)]
        for i in range(SRC_NUM):
            src_demean[i][0] = src[2 * i] - src_mean[0]
            src_demean[i][1] = src[2 * i + 1] - src_mean[1]
            dst_demean[i][0] = self.umeyama_args_112[2 * i] - dst_mean[0]
            dst_demean[i][1] = self.umeyama_args_112[2 * i + 1] - dst_mean[1]

        # 计算A矩阵 / Calculate A matrix
        A = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(SRC_DIM):
            for k in range(SRC_DIM):
                for j in range(SRC_NUM):
                    A[i][k] += dst_demean[j][i] * src_demean[j][k]
                A[i][k] /= SRC_NUM

        # SVD分解和旋转矩阵计算 / SVD decomposition and rotation matrix calculation
        T = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        U, S, V = self.svd22([A[0][0], A[0][1], A[1][0], A[1][1]])
        T[0][0] = U[0] * V[0] + U[1] * V[2]
        T[0][1] = U[0] * V[1] + U[1] * V[3]
        T[1][0] = U[2] * V[0] + U[3] * V[2]
        T[1][1] = U[2] * V[1] + U[3] * V[3]

        # 计算缩放因子 / Calculate scaling factor
        scale = 1.0
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

        # 计算平移向量 / Calculate translation vector
        T[0][2] = dst_mean[0] - scale * (T[0][0] * src_mean[0] + T[0][1] * src_mean[1])
        T[1][2] = dst_mean[1] - scale * (T[1][0] * src_mean[0] + T[1][1] * src_mean[1])

        # 应用缩放 / Apply scaling
        T[0][0] *= scale
        T[0][1] *= scale
        T[1][0] *= scale
        T[1][1] *= scale

        return T

    def get_affine_matrix(self, sparse_points):
        """
        获取仿射变换矩阵 / Get affine transformation matrix
        """
        with ScopedTiming("get_affine_matrix", self.debug_mode > 1):
            matrix_dst = self.image_umeyama_112(sparse_points)
            matrix_dst = [matrix_dst[0][0], matrix_dst[0][1], matrix_dst[0][2],
                         matrix_dst[1][0], matrix_dst[1][1], matrix_dst[1][2]]
            return matrix_dst

class FaceRecognition:
    """
    人脸识别类 / Face recognition class
    集成了检测和识别功能 / Integrates detection and recognition functions
    """
    def __init__(self, face_det_kmodel, face_reg_kmodel, det_input_size, reg_input_size,
                 database_dir, anchors, confidence_threshold=0.25, nms_threshold=0.3,
                 face_recognition_threshold=0.75, rgb888p_size=[1280,720],
                 display_size=[640,360], debug_mode=0):

        # 初始化参数 / Initialize parameters
        self.face_det_kmodel = face_det_kmodel
        self.face_reg_kmodel = face_reg_kmodel
        self.det_input_size = det_input_size
        self.reg_input_size = reg_input_size
        self.database_dir = database_dir
        self.anchors = anchors
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.face_recognition_threshold = face_recognition_threshold
        self.rgb888p_size = [ALIGN_UP(rgb888p_size[0],16), rgb888p_size[1]]
        self.display_size = [ALIGN_UP(display_size[0],16), display_size[1]]
        self.debug_mode = debug_mode

        # 数据库参数 / Database parameters
        self.max_register_face = 100
        self.feature_num = 128
        self.valid_register_face = 0
        self.db_name = []
        self.db_data = []

        # 初始化检测和注册模型 / Initialize detection and registration models
        self.face_det = FaceDetApp(self.face_det_kmodel,
                                 model_input_size=self.det_input_size,
                                 anchors=self.anchors,
                                 confidence_threshold=self.confidence_threshold,
                                 nms_threshold=self.nms_threshold,
                                 rgb888p_size=self.rgb888p_size,
                                 display_size=self.display_size,
                                 debug_mode=0)

        self.face_reg = FaceRegistrationApp(self.face_reg_kmodel,
                                          model_input_size=self.reg_input_size,
                                          rgb888p_size=self.rgb888p_size,
                                          display_size=self.display_size)

        self.face_det.config_preprocess()
        self.database_init()

    def run(self, input_np):
        """
        运行人脸识别 / Run face recognition
        """
        # 人脸检测 / Face detection
        det_boxes, landms = self.face_det.run(input_np)
        recg_res = []

        # 对每个检测到的人脸进行识别 / Recognize each detected face
        for landm in landms:
            self.face_reg.config_preprocess(landm)
            feature = self.face_reg.run(input_np)
            res = self.database_search(feature)
            recg_res.append(res)

        return det_boxes, recg_res

    def database_init(self):
        """
        初始化人脸数据库 / Initialize face database
        """
        with ScopedTiming("database_init", self.debug_mode > 1):
            try:
                # 读取数据库文件 / Read database files
                db_file_list = os.listdir(self.database_dir)
                for db_file in db_file_list:
                    if not db_file.endswith('.bin'):
                        continue
                    if self.valid_register_face >= self.max_register_face:
                        break
                        
                    valid_index = self.valid_register_face
                    full_db_file = self.database_dir + db_file
                    
                    # 读取特征数据 / Read feature data
                    with open(full_db_file, 'rb') as f:
                        data = f.read()
                    feature = np.frombuffer(data, dtype=np.float)
                    self.db_data.append(feature)
                    
                    # 保存人名 / Save person name
                    name = db_file.split('.')[0]
                    self.db_name.append(name)
                    self.valid_register_face += 1
            except Exception as e:
                print(e)
                print("未检测到人脸数据库，请先按照教程步骤，注册人脸信息")
                print("No face database detected, please follow the tutorial steps to register the face information")
            # 读取数据库文件 / Read database files
            db_file_list = os.listdir(self.database_dir)
            for db_file in db_file_list:
                if not db_file.endswith('.bin'):
                    continue
                if self.valid_register_face >= self.max_register_face:
                    break

                valid_index = self.valid_register_face
                full_db_file = self.database_dir + db_file

                # 读取特征数据 / Read feature data
                with open(full_db_file, 'rb') as f:
                    data = f.read()
                feature = np.frombuffer(data, dtype=np.float)
                self.db_data.append(feature)

                # 保存人名 / Save person name
                name = db_file.split('.')[0]
                self.db_name.append(name)
                self.valid_register_face += 1

    def database_reset(self):
        """
        重置数据库 / Reset database
        """
        with ScopedTiming("database_reset", self.debug_mode > 1):
            print("database clearing...")
            self.db_name = []
            self.db_data = []
            self.valid_register_face = 0
            print("database clear Done!")

    def database_search(self, feature):
        """
        在数据库中搜索匹配的人脸 / Search for matching face in database
        """
        with ScopedTiming("database_search", self.debug_mode > 1):
            v_id = -1
            v_score_max = 0.0

            # 特征归一化 / Feature normalization
            feature /= np.linalg.norm(feature)

            # 遍历数据库进行匹配 / Search through database for matches
            for i in range(self.valid_register_face):
                db_feature = self.db_data[i]
                db_feature /= np.linalg.norm(db_feature)
                v_score = np.dot(feature, db_feature)/2 + 0.5

                if v_score > v_score_max:
                    v_score_max = v_score
                    v_id = i

            # 返回识别结果 / Return recognition result
            if v_id == -1:
                return 'unknown'
            elif v_score_max < self.face_recognition_threshold:
                return 'unknown'
            else:
                result = 'name: {}, score: {}'.format(self.db_name[v_id], v_score_max)
                return result

    def draw_result(self, pl, dets, recg_results):
        """
        绘制识别结果 / Draw recognition results
        """
        pl.osd_img.clear()
        if dets:
            for i, det in enumerate(dets):
                # 绘制人脸框 / Draw face box
                x1, y1, w, h = map(lambda x: int(round(x, 0)), det[:4])
                x1 = x1 * self.display_size[0]//self.rgb888p_size[0]
                y1 = y1 * self.display_size[1]//self.rgb888p_size[1]
                w = w * self.display_size[0]//self.rgb888p_size[0]
                h = h * self.display_size[1]//self.rgb888p_size[1]

                # 绘制识别结果 / Draw recognition result
                recg_text = recg_results[i]
                if recg_text == 'unknown':
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,0,255), thickness=4)
                else:
                    pl.osd_img.draw_rectangle(x1, y1, w, h, color=(255,0,255,0), thickness=4)
                pl.osd_img.draw_string_advanced(x1, y1, 32, recg_text, color=(255,255,0,0))

                # 使用正则表达式匹配 name 和 score 的值
                pattern = r'name: (.*), score: (.*)'
                match = re.match(pattern, recg_text)

                if match:
                    name_value = match.group(1)  # 提取 name 的值
                    score_value = match.group(2)  # 提取 score 的值
                    pto_data = pto.get_face_recoginiton_data(x1, y1, w, h, name_value, score_value)
                    uart.send(pto_data)
                    print(pto_data)
                else:
                    pto_data = pto.get_face_recoginiton_data(x1, y1, w, h, recg_text, 0)
                    uart.send(pto_data)
                    print(pto_data)


def exce_demo(pl):
    """
    执行演示程序 / Execute demo program
    """
    global fr
    display_mode = pl.display_mode
    rgb888p_size = pl.rgb888p_size
    display_size = pl.display_size

    # 加载模型和配置 / Load models and configurations
    face_det_kmodel_path = "/sdcard/kmodel/face_detection_320.kmodel"
    face_reg_kmodel_path = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    database_dir = "/data/face_database/2600271ef6d/"
    face_det_input_size = [320,320]
    face_reg_input_size = [112,112]
    confidence_threshold = 0.5
    nms_threshold = 0.2
    anchor_len = 4200
    det_dim = 4

    # 读取anchor数据 / Read anchor data
    anchors = np.fromfile(anchors_path, dtype=np.float)
    anchors = anchors.reshape((anchor_len, det_dim))
    face_recognition_threshold = 0.65

    # 创建人脸识别对象 / Create face recognition object
    fr = FaceRecognition(face_det_kmodel_path, face_reg_kmodel_path,
                        det_input_size=face_det_input_size,
                        reg_input_size=face_reg_input_size,
                        database_dir=database_dir,
                        anchors=anchors,
                        confidence_threshold=confidence_threshold,
                        nms_threshold=nms_threshold,
                        face_recognition_threshold=face_recognition_threshold,
                        rgb888p_size=rgb888p_size,
                        display_size=display_size)

    # 主循环 / Main loop
    try:
        while True:
            with ScopedTiming("total", 1):
                # 获取图像并处理 / Get and process image
                img = pl.get_frame()
                det_boxes, recg_res = fr.run(img)
                fr.draw_result(pl, det_boxes, recg_res)
                pl.show_image()
                gc.collect()
    except Exception as e:
        print("人脸识别功能退出")
    finally:
        exit_demo()

def exit_demo():
    """
    退出程序 / Exit program
    """
    global fr
    fr.face_det.deinit()
    fr.face_reg.deinit()

if __name__ == "__main__":
    # 主程序入口 / Main program entry
    rgb888p_size=[640,480]
    display_size = [640,480]
    display_mode = "lcd"

    # 创建并启动视频处理Pipeline / Create and start video processing pipeline
    pl = PipeLine(rgb888p_size=rgb888p_size,
                 display_size=display_size,
                 display_mode=display_mode)
    pl.create()
    exce_demo(pl)
```
控制端相关例程
```PYTHON
import serial
​
com="/dev/ttyUSB0"
ser = serial.Serial(com, 115200)
​
FUNC_ID = 6
​
def parse_data(data):
    if data[0] == ord('$') and data[len(data)-1] == ord('#'):
        data_list = data[1:len(data)-1].decode('utf-8').split(',')
        data_len = int(data_list[0])
        data_id = int(data_list[1])
        if data_len == len(data) and data_id == FUNC_ID:
            # print(data_list)
            x = int(data_list[2])
            y = int(data_list[3])
            w = int(data_list[4])
            h = int(data_list[5])
            return x, y, w, h
        elif (data_len != len(data)):
            print("data len error:", data_len, len(data))
        elif(data_id != FUNC_ID):
            print("func id error:", data_id, FUNC_ID)
    else:
        print("pto error", data)
    return -1, -1, -1, -1
​
while True:
    if ser.in_waiting:
        data = ser.readline()
        # print("rx:", data)
        x, y, w, h = parse_data(data.rstrip(b'\n'))
        print("face:x:%d, y:%d, w:%d, h:%d" % (x, y, w, h))
​```
人脸识别（控制端）
```PYTHON
import serial
import time
​
com="/dev/ttyUSB0"
ser = serial.Serial(com, 115200)
​
FUNC_ID = 8
​
​
def parse_data(data):
    if data[0] == ord('$') and data[len(data)-1] == ord('#'):
        data_list = data[1:len(data)-1].decode('utf-8').split(',')
        data_len = int(data_list[0])
        data_id = int(data_list[1])
        if data_len == len(data) and data_id == FUNC_ID:
            # print(data_list)
            x = int(data_list[2])
            y = int(data_list[3])
            w = int(data_list[4])
            h = int(data_list[5])
            msg = data_list[6]
            score = int(data_list[7])
            return x, y, w, h, msg, score
        elif (data_len != len(data)):
            print("data len error:", data_len, len(data))
        elif(data_id != FUNC_ID):
            print("func id error:", data_id, FUNC_ID)
    return -1, -1, -1, -1, "", -1
​
while True:
    if ser.in_waiting:
        data = ser.readline()
        # print("rx:", data)
        x, y, w, h, msg, score = parse_data(data.rstrip(b'\n'))
        print("face recogition:x:%d, y:%d, w:%d, h:%d, msg:%s, score:%d" % (x, y, w, h, msg, score))
```
人脸注册没有相关控制端例程

这是相关可能用得到的API手册

UART 模块 API 手册
概述
K230 内部集成了五个 UART（通用异步收发传输器）硬件模块，其中 UART0 被小核 SH 占用，UART3 被大核 SH 占用，剩余的 UART1、UART2 和 UART4 可供用户使用。UART 的 I/O 配置可参考 IOMUX 模块。

API 介绍
UART 类位于 machine 模块中。

示例代码
from machine import UART

# 配置 UART1: 波特率 115200, 8 位数据位, 无奇偶校验, 1 个停止位
u1 = UART(UART.UART1, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)

# 写入数据到 UART
u1.write("UART1 test")

# 从 UART 读取数据
r = u1.read()

# 读取一行数据
r = u1.readline()

# 将数据读入字节缓冲区
b = bytearray(8)
r = u1.readinto(b)

# 释放 UART 资源
u1.deinit()
构造函数
uart = UART(id, baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE, timeout = 0)
参数

id: UART 模块编号，有效值为 UART1、UART2、UART4。

baudrate: UART 波特率，可选参数，默认值为 115200。

bits: 每个字符的数据位数，有效值为 FIVEBITS、SIXBITS、SEVENBITS、EIGHTBITS，可选参数，默认值为 EIGHTBITS。

parity: 奇偶校验，有效值为 PARITY_NONE、PARITY_ODD、PARITY_EVEN，可选参数，默认值为 PARITY_NONE。

stop: 停止位数，有效值为 STOPBITS_ONE、STOPBITS_TWO，可选参数，默认值为 STOPBITS_ONE。

timeout: 读数据超时，单位为 ms

init 方法
UART.init(baudrate=115200, bits=UART.EIGHTBITS, parity=UART.PARITY_NONE, stop=UART.STOPBITS_ONE)
配置 UART 参数。

参数

参考构造函数。

返回值

无

read 方法
UART.read([nbytes])
读取字符。如果指定了 nbytes，则最多读取该数量的字节；否则，将尽可能多地读取数据。

参数

nbytes: 最多读取的字节数，可选参数。

返回值

返回一个包含读取字节的字节对象。

readline 方法
UART.readline()
读取一行数据，并以换行符结束。

参数

无

返回值

返回一个包含读取字节的字节对象。

readinto 方法
UART.readinto(buf[, nbytes])
将字节读取到 buf 中。如果指定了 nbytes，则最多读取该数量的字节；否则，最多读取 len(buf) 数量的字节。

参数

buf: 一个缓冲区对象。

nbytes: 最多读取的字节数，可选参数。

返回值

返回读取并存入 buf 的字节数。

write 方法
UART.write(buf)
将字节缓冲区写入 UART。

参数

buf: 一个缓冲区对象。

返回值

返回写入的字节数。

deinit 方法
UART.deinit()
释放 UART 资源。

参数

无

通信协议：无文档 但是我在源码中找到如下：


class YbProtocol:
    def __init__(self):
        self.ID_COLOR = 1
        self.ID_BARCODE = 2
        self.ID_QRCODE = 3
        self.ID_APRILTAG = 4
        self.ID_DMCODE = 5
        self.ID_FACE_DETECT = 6
        self.ID_EYE_GAZE = 7
        self.ID_FACE_RECOGNITION = 8
        self.ID_PERSON_DETECT = 9
        self.ID_FALLDOWN_DETECT = 10
        self.ID_HAND_DETECT = 11
        self.ID_HAND_GESTURE = 12
        self.ID_OCR_REC = 13
        self.ID_OBJECT_DETECT = 14
        self.ID_NANO_TRACKER = 15
        self.ID_SELF_LEARNING = 16
        self.ID_LICENCE_REC = 17
        self.ID_LICENCE_DETECT = 18
        self.ID_GARBAGE_DETECT = 19
        self.ID_GUIDE_DETECT = 20
        self.ID_OBSTACLE_DETECT = 21
        self.ID_MULTI_COLOR = 22
        self.ID_FINGER_GUESS = 23

        

    def package_coord(self, func, x, y, w, h, msg=None):
        pto_len = 0
        if msg is None:
            temp_buf = "$%02d,%02d,%03d,%03d,%03d,%03d#" % (pto_len, func, x, y, w, h)
            pto_len = len(temp_buf)
            pto_buf = "$%02d,%02d,%03d,%03d,%03d,%03d#\n" % (pto_len, func, x, y, w, h)
        else:
            temp_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%s#" % (pto_len, func, x, y, w, h, msg)
            pto_len = len(temp_buf)
            pto_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%s#\n" % (pto_len, func, x, y, w, h, msg)
        return pto_buf
    

    def package_message(self, func, msg, value=None):
        pto_len = 0
        if value is None:
            temp_buf = "$%02d,%02d,%s#" % (pto_len, func, msg)
            pto_len = len(temp_buf)
            pto_buf = "$%02d,%02d,%s#\n" % (pto_len, func, msg)
        else:
            temp_buf = "$%02d,%02d,%s,%03d#" % (pto_len, func, msg, value)
            pto_len = len(temp_buf)
            pto_buf = "$%02d,%02d,%s,%03d#\n" % (pto_len, func, msg, value)
        return pto_buf

    def package_msg_value(self, func, x, y, w, h, msg, value):
        pto_len = 0
        temp_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%s,%03d#" % (pto_len, func, x, y, w, h, msg, value)
        pto_len = len(temp_buf)
        pto_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%s,%03d#\n" % (pto_len, func, x, y, w, h, msg, value)
        return pto_buf
    
    def package_apriltag(self, func, x, y, w, h, tag_id, degrees):
        pto_len = 0
        temp_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%03d,%03d#" % (pto_len, func, x, y, w, h, tag_id, degrees)
        pto_len = len(temp_buf)
        pto_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%03d,%03d#\n" % (pto_len, func, x, y, w, h, tag_id, degrees)
        return pto_buf
    
    def package_licence(self, func, msg):
        pto_len = 0
        temp_buf = "$%02d,%02d,%s#" % (pto_len, func, msg)
        pto_len = len(temp_buf) + 2
        pto_buf = "$%02d,%02d,%s#\n" % (pto_len, func, msg)
        return pto_buf

    def package_point8(self, func, point8):
        pto_len = 0
        if (len(point8) != 8):
            return
        temp_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%03d,%03d,%03d,%03d#" % (pto_len, func, point8[0], point8[1], point8[2], point8[3], point8[4], point8[5], point8[6], point8[7])
        pto_len = len(temp_buf)
        pto_buf = "$%02d,%02d,%03d,%03d,%03d,%03d,%03d,%03d,%03d,%03d#\n" % (pto_len, func, point8[0], point8[1], point8[2], point8[3], point8[4], point8[5], point8[6], point8[7])
        return pto_buf



    #########################################################################################################
    #########################################################################################################
    #########################################################################################################
    
    def get_color_data(self, x, y, w, h):
        func_id = self.ID_COLOR
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h))
        return data
    
    def get_barcode_data(self, x, y, w, h, msg):
        func_id = self.ID_BARCODE
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_qrcode_data(self, x, y, w, h, msg):
        func_id = self.ID_QRCODE
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_apriltag_data(self, x, y, w, h, tag_id, degrees):
        func_id = self.ID_APRILTAG
        data = self.package_apriltag(func_id, int(x), int(y), int(w), int(h), int(tag_id), int(degrees))
        return data
    
    def get_dmcode_data(self, x, y, w, h, msg, degrees):
        func_id = self.ID_DMCODE
        data = self.package_msg_value(func_id, int(x), int(y), int(w), int(h), msg, int(degrees))
        return data

    def get_face_detect_data(self, x, y, w, h):
        func_id = self.ID_FACE_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h))
        return data
    
    def get_eye_gaze_data(self, start_x, start_y, end_x, end_y):
        func_id = self.ID_EYE_GAZE
        data = self.package_coord(func_id, int(start_x), int(start_y), int(end_x), int(end_y))
        return data
    
    def get_face_recoginiton_data(self, x, y, w, h, name, score):
        func_id = self.ID_FACE_RECOGNITION
        data = self.package_msg_value(func_id, int(x), int(y), int(w), int(h), name, int(float(score)*100))
        return data
    
    def get_person_detect_data(self, x, y, w, h):
        func_id = self.ID_PERSON_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h))
        return data

    def get_falldown_detect_data(self, x, y, w, h, msg, score):
        func_id = self.ID_FALLDOWN_DETECT
        data = self.package_msg_value(func_id, int(x), int(y), int(w), int(h), msg, int(float(score)*100))
        return data


    def get_hand_detect_data(self, x, y, w, h):
        func_id = self.ID_HAND_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h))
        return data


    def get_hand_gesture_data(self, msg):
        func_id = self.ID_HAND_GESTURE
        data = self.package_message(func_id, msg)
        return data

    def get_ocr_rec_data(self, msg):
        func_id = self.ID_OCR_REC
        data = self.package_message(func_id, msg)
        return data

    def get_object_detect_data(self, x, y, w, h, msg):
        func_id = self.ID_OBJECT_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data

    def get_nano_tracker_data(self, x, y, w, h):
        func_id = self.ID_NANO_TRACKER
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h))
        return data

    def get_self_learning_data(self, category, score):
        func_id = self.ID_SELF_LEARNING
        data = self.package_message(func_id, category, int(float(score)*100))
        return data

    def get_licence_rec_data(self, msg):
        func_id = self.ID_LICENCE_REC
        data = self.package_licence(func_id, msg)
        return data

    def get_licence_detect_data(self, point8):
        func_id = self.ID_LICENCE_DETECT
        data = self.package_point8(func_id, point8)
        return data

    def get_garbage_detect_data(self, x, y, w, h, msg):
        func_id = self.ID_GARBAGE_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_guide_detect_data(self, x, y, w, h, msg):
        func_id = self.ID_GUIDE_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_obstacle_detect_data(self, x, y, w, h, msg):
        func_id = self.ID_OBSTACLE_DETECT
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_multi_color_data(self, x, y, w, h, msg):
        func_id = self.ID_MULTI_COLOR
        data = self.package_coord(func_id, int(x), int(y), int(w), int(h), msg)
        return data
    
    def get_finger_guess_data(self, msg):
        func_id = self.ID_FINGER_GUESS
        data = self.package_message(func_id, msg)
        return data


无需WiFi通信

多线程：
多线程模块 (_thread)
_thread是MicroPython的一个基本线程模块,用于实现多线程编程

这里有几点是在使用过程中需要着重注意的

_thread 模块的多线程实现的是系统级别的多线程

这意味着程序从硬件的角度上来看其实还是单线程进行的，只是Micropython通过内部的线程调度机制

模拟的实现多线程的效果。

_thread 模块的线程调度机制是非抢占式的

这意味着你必须手动的去避免某一个线程一直占用处理器资源，通常的做法是在每个线程（如果有循环的话）

在循环的结尾处添加一个主动的延迟函数time.sleep_us(1)。只有当执行到sleep的时候，系统才会进行一次线程的调度

python的_thread模块经过了多次的更新，在3.8以前都未被正式的确定使用，

K230固件中的Micropython对_thread的使用方法并不一定完全符合最新版python中的文档

类型
LockType
线程锁类型，用于线程同步。

方法:
acquire(): 获取锁。如果锁已被其他线程持有，将阻塞直到锁被释放。
locked(): 返回锁的状态。如果锁被某个线程持有返回 True，否则返回 False。
release(): 释放锁。只能由持有锁的线程调用。
函数
allocate_lock()
创建并返回一个新的锁对象。

exit()
终止调用它的线程。

get_ident()
返回当前线程的标识符。

stack_size([size])
设置或获取新创建线程的栈大小。

start_new_thread(function, args)
启动一个新线程，执行给定的函数，并传入指定的参数元组。

Timer 模块 API 手册
概述
K230 内部集成了 6 个 Timer 硬件模块，最小定时周期为 1 毫秒（ms）。

API 介绍
Timer 类位于 machine 模块中。

示例代码
from machine import Timer
import time

# 实例化一个软定时器
tim = Timer(-1)

# 配置定时器，单次模式，周期 100 毫秒，回调函数打印 1
tim.init(period=100, mode=Timer.ONE_SHOT, callback=lambda t: print(1))
time.sleep(0.2)

# 配置定时器，周期模式，周期 1000 毫秒，回调函数打印 2
tim.init(freq=1, mode=Timer.PERIODIC, callback=lambda t: print(2))
time.sleep(2)

# 释放定时器资源
tim.deinit()
构造函数
timer = Timer(index, mode=Timer.PERIODIC, freq=-1, period=-1, callback=None)
参数

index: Timer 模块编号，取值范围为 [-1, 5]，其中 -1 表示软件定时器。

mode: 定时器运行模式，可以是单次或周期模式（可选参数）。

freq: 定时器运行频率，支持浮点数，单位为赫兹（Hz），此参数优先级高于 period（可选参数）。

period: 定时器运行周期，单位为毫秒（ms）（可选参数）。

callback: 超时回调函数，必须设置并应带有一个参数。

init 方法
Timer.init(mode=Timer.PERIODIC, freq=-1, period=-1, callback=None)
初始化定时器参数。

参数

mode: 定时器运行模式，可以是单次或周期模式（可选参数）。

freq: 定时器运行频率，支持浮点数，单位为赫兹（Hz），此参数优先级高于 period（可选参数）。

period: 定时器运行周期，单位为毫秒（ms）（可选参数）。

callback: 超时回调函数，必须设置并应带有一个参数。

返回值

无

deinit 方法
Timer.deinit()
释放定时器资源。

参数

无

返回值

无

Pin 模块 API 手册
概述
K230 芯片内部包含 64 个 GPIO（通用输入输出）引脚，每个引脚均可配置为输入或输出模式，并支持上下拉电阻配置和驱动能力设置。这些引脚能够灵活用于各种数字输入输出场景。

API 介绍
Pin 类位于 machine 模块中，用于控制 K230 芯片的 GPIO 引脚。

示例

from machine import Pin

# 将引脚 2 配置为输出模式，无上下拉，驱动能力为 7
pin = Pin(2, Pin.OUT, pull=Pin.PULL_NONE, drive=7)

# 设置引脚 2 输出高电平
pin.value(1)

# 设置引脚 2 输出低电平
pin.value(0)
构造函数
pin = Pin(index, mode, pull=Pin.PULL_NONE, value = -1, drive=7, alt = -1)
参数

index: 引脚编号，范围为 [0, 63]。

mode: 引脚的模式，支持输入模式或输出模式。

pull: 上下拉配置（可选），默认为 Pin.PULL_NONE。

drive: 驱动能力配置（可选），默认值为 7。

value: 设置引脚默认输出值

alt: 目前未使用

init 方法
Pin.init(mode, pull=Pin.PULL_NONE, drive=7)
用于初始化引脚的模式、上下拉配置及驱动能力。

参数

mode: 引脚的模式（输入或输出）。

pull: 上下拉配置（可选），默认值为 Pin.PULL_NONE。

drive: 驱动能力（可选），默认值为 7。

返回值

无

value 方法
Pin.value([value])
获取引脚的输入电平值或设置引脚的输出电平。

参数

value: 输出值（可选），如果传递该参数则设置引脚输出为指定值。如果不传参则返回引脚的当前输入电平值。

返回值

返回空或当前引脚的输入电平值。

mode 方法
Pin.mode([mode])
获取或设置引脚的模式。

参数

mode: 引脚模式（输入或输出），如果不传参则返回当前引脚的模式。

返回值

返回空或当前引脚模式。

pull 方法
Pin.pull([pull])
获取或设置引脚的上下拉配置。

参数

pull: 上下拉配置（可选），如果不传参则返回当前上下拉配置。

返回值

返回空或当前引脚的上下拉配置。

drive 方法
Pin.drive([drive])
获取或设置引脚的驱动能力。

参数

drive: 驱动能力（可选），如果不传参则返回当前驱动能力。

返回值

返回空或当前引脚的驱动能力。

on 方法
Pin.on()
将引脚输出设置为高电平。

参数

无

返回值

无

off 方法
Pin.off()
将引脚输出设置为低电平。

参数

无

返回值

无

high 方法
Pin.high()
将引脚输出设置为高电平。

参数

无

返回值

无

low 方法
Pin.low()
将引脚输出设置为低电平。

参数

无

返回值

无

irq 方法
Pin.irq(handler=None, trigger=Pin.IRQ_FALLING | Pin.IRQ_RISING, *, priority=1, wake=None, hard=False, debounce = 10)
使能 IO 中断功能

handler: 回调函数，必须设置

trigger: 触发模式

priority: 不支持

wake: 不支持

hard: 不支持

debounce: 高电平和低电平触发时，最小触发间隔，单位为 ms，最小值为 5

返回值

mq_irq 对象

常量定义
模式
Pin.IN: 输入模式

Pin.OUT: 输出模式

上下拉模式
PULL_NONE: 关掉上下拉

PULL_UP: 使能上拉

PULL_DOWN: 使能下拉

中断触发模式
IRQ_FALLING: 下降沿触发

IRQ_RISING: 上升沿触发

IRQ_LOW_LEVEL: 低电平触发

IRQ_HIGH_LEVEL: 高电平触发

IRQ_BOTH: 边沿触发

驱动能力
具体配置对应的电流输出能力参见fpioa

DRIVE_0

DRIVE_1

DRIVE_2

DRIVE_3

DRIVE_4

DRIVE_5

DRIVE_6

DRIVE_7

DRIVE_8

DRIVE_9

DRIVE_10

DRIVE_11

DRIVE_12

DRIVE_13

DRIVE_14

DRIVE_15

需要双向控制
只用串口通信
树莓派的后面再说 先把K230搞好
