"""
人脸注册功能模块
"""

from libs.PipeLine import ScopedTiming
from libs.AIBase import AIBase
from libs.AI2D import Ai2d
from media.media import *
import nncase_runtime as nn
import ulab.numpy as np
import aidemo
import image
import os
import gc


def ensure_dir(directory):
    """递归创建目录"""
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
                ensure_dir(parent)
        
        try:
            os.mkdir(directory)
            print("[Register] Created dir:", directory)
        except:
            pass


def face_register_handler(controller, user_id, photo_path):
    """
    人脸注册处理函数
    
    参数:
        controller: 控制器实例
        user_id: 用户ID
        photo_path: 照片路径
    
    返回:
        (success, message)
    """
    from ai_modules.face_recognition import FaceDetApp, FaceFeatureApp
    
    kmodel_det = "/sdcard/kmodel/face_detection_320.kmodel"
    kmodel_reg = "/sdcard/kmodel/face_recognition.kmodel"
    anchors_path = "/sdcard/utils/prior_data_320.bin"
    
    face_det = None
    face_feature = None
    
    try:
        # 检查照片是否存在
        try:
            os.stat(photo_path)
        except:
            return False, "Photo not found: %s" % photo_path
        
        # 加载anchors
        anchors = np.fromfile(anchors_path, dtype=np.float)
        anchors = anchors.reshape((4200, 4))
        
        # 读取图片
        img = image.Image(photo_path)
        img_data = img.to_rgb888()
        img_hwc = img_data.to_numpy_ref()
        shape = img_hwc.shape
        
        # 转换为NCHW格式
        img_tmp = img_hwc.reshape((shape[0] * shape[1], shape[2]))
        img_trans = img_tmp.transpose()
        img_res = img_trans.copy()
        input_np = img_res.reshape((1, shape[2], shape[0], shape[1]))
        
        input_size = [input_np.shape[3], input_np.shape[2]]
        
        # 初始化检测器
        face_det = FaceDetApp(
            kmodel_det,
            model_input_size=[320, 320],
            anchors=anchors,
            confidence_threshold=0.5,
            nms_threshold=0.2,
            rgb888p_size=input_size,
            display_size=input_size
        )
        face_det.config_preprocess(input_size)
        
        # 初始化特征提取器
        face_feature = FaceFeatureApp(
            kmodel_reg,
            model_input_size=[112, 112],
            rgb888p_size=input_size,
            display_size=input_size
        )
        
        # 检测人脸
        det_boxes, landms = face_det.run(input_np)
        
        if det_boxes is None or len(det_boxes) == 0:
            return False, "No face detected"
        
        if len(det_boxes) > 1:
            return False, "Multiple faces detected, need exactly one"
        
        # 提取特征
        face_feature.config_preprocess(landms[0], input_size)
        feature = face_feature.run(input_np)
        
        # 保存到数据库
        database_dir = controller.config.get('database_dir', '/data/face_database/')
        ensure_dir(database_dir)
        
        feature_path = database_dir + user_id + ".bin"
        with open(feature_path, "wb") as f:
            f.write(feature.tobytes())
        
        print("[Register] Saved:", feature_path)
        
        return True, "Registered:%s" % user_id
        
    except Exception as e:
        return False, str(e)
    
    finally:
        if face_det:
            face_det.deinit()
        if face_feature:
            face_feature.deinit()
        gc.collect()