"""
K230 媒体资源管理助手
修复：使用正确的 MediaManager API
"""

from media.media import *
import time
import gc

# 调试开关
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print("[MediaHelper]", *args)


class MediaHelper:
    """
    媒体资源管理器
    
    K230 资源释放的正确顺序（根据官方文档）：
    1. 停止帧处理循环
    2. 调用 sensor.stop()（必须在 MediaManager.deinit() 之前）
    3. 调用 MediaManager.deinit()
    4. 释放 AI 模型
    5. 强制 GC
    """
    
    # 类变量：追踪当前活跃的资源
    _active_pipeline = None
    _active_sensor = None
    _active_models = []
    
    @classmethod
    def destroy_pipeline_safe(cls, pipeline):
        """
        安全销毁 Pipeline
        必须在释放模型之前调用！
        """
        debug_print("=== destroy_pipeline_safe ===")
        
        if pipeline is None:
            debug_print("Pipeline is None, skip")
            return
        
        try:
            debug_print("Calling pipeline.destroy()...")
            pipeline.destroy()
            debug_print("Pipeline destroyed")
        except Exception as e:
            debug_print("Pipeline destroy error:", e)
            # 尝试手动清理
            cls._manual_cleanup()
        
        cls._active_pipeline = None
        cls._active_sensor = None
        
        # 等待媒体系统停止
        debug_print("Waiting for media to stop...")
        time.sleep(0.3)
    
    @classmethod
    def release_models_safe(cls, models):
        """
        安全释放模型列表
        必须在 Pipeline 销毁之后调用！
        """
        debug_print("=== release_models_safe ===")
        
        if models is None:
            return
        
        # 统一处理为列表
        if not isinstance(models, (list, tuple)):
            models = [models]
        
        for i, model in enumerate(models):
            if model is None:
                continue
            
            try:
                debug_print("Releasing model %d..." % i)
                model.deinit()
                debug_print("Model %d released" % i)
            except Exception as e:
                debug_print("Model %d release error:" % i, e)
        
        cls._active_models.clear()
    
    @classmethod
    def cleanup_all(cls, pipeline=None, models=None, wait_time=0.5):
        """
        完整清理：Pipeline + 模型
        """
        debug_print("=== cleanup_all START ===")
        
        # 步骤1：先销毁 Pipeline（停止媒体流）
        cls.destroy_pipeline_safe(pipeline)
        
        # 步骤2：释放模型
        cls.release_models_safe(models)
        
        # 步骤3：强制 GC
        debug_print("Force gc.collect()...")
        gc.collect()
        
        # 步骤4：最后等待
        if wait_time > 0:
            debug_print("Final wait %.2fs..." % wait_time)
            time.sleep(wait_time)
        
        debug_print("=== cleanup_all DONE ===")
    
    @classmethod
    def _manual_cleanup(cls):
        """手动清理媒体资源"""
        debug_print("=== Manual cleanup ===")
        
        # 尝试调用 MediaManager.deinit()
        try:
            MediaManager.deinit()
            debug_print("MediaManager.deinit() called")
        except Exception as e:
            debug_print("MediaManager.deinit() error:", e)
        
        gc.collect()
        time.sleep(0.3)
    
    @classmethod
    def force_reset(cls):
        """
        强制重置所有媒体资源
        当系统卡住时使用
        """
        debug_print("=== FORCE RESET ===")
        
        # 清空追踪
        cls._active_pipeline = None
        cls._active_sensor = None
        cls._active_models.clear()
        
        # 尝试调用 MediaManager.deinit()
        try:
            MediaManager.deinit()
            debug_print("MediaManager.deinit() success")
        except Exception as e:
            debug_print("MediaManager.deinit() error:", e)
        
        # 多次 GC
        for i in range(3):
            gc.collect()
            time.sleep(0.2)
        
        debug_print("=== FORCE RESET DONE ===")
    
    @classmethod
    def register_pipeline(cls, pipeline):
        """注册活跃的 Pipeline"""
        cls._active_pipeline = pipeline
    
    @classmethod
    def register_model(cls, model):
        """注册活跃的模型"""
        if model not in cls._active_models:
            cls._active_models.append(model)


def safe_cleanup(pipeline=None, models=None):
    """便捷清理函数"""
    MediaHelper.cleanup_all(pipeline, models)


def force_reset():
    """强制重置媒体系统"""
    MediaHelper.force_reset()