"""
K230 摄像头管理器
24小时常驻运行，提供帧获取接口
支持动态帧率调节
"""

from media.sensor import *
from media.display import *
from media.media import *
import time
import gc
import image

class CameraManager:
    """
    摄像头管理器 - 单例模式
    """
    
    _instance = None
    
    FPS_LEVELS = {
        'high': 30,
        'medium': 15,
        'low': 10,
        'idle': 5
    }
    
    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = CameraManager()
        return cls._instance
    
    def __init__(self):
        if CameraManager._instance is not None:
            raise Exception("Use CameraManager.get_instance()")
        
        # 配置参数
        self.rgb888p_size = [640, 480]
        self.display_size = [640, 480]
        
        # 状态
        self._running = False
        self._paused = False
        self._current_fps = 30
        
        # 媒体对象
        self._sensor = None
        self._osd_img = None
        self._media_initialized = False
        
        # 帧率控制
        self._frame_interval_ms = 33
        self._last_frame_time = 0
        
        print("[Camera] Manager created")
    
    def start(self, fps=30):
        """启动摄像头"""
        if self._running:
            print("[Camera] Already running")
            return True
        
        print("[Camera] Starting...")
        
        try:
            # 对齐尺寸
            self.rgb888p_size[0] = self._align_up(self.rgb888p_size[0], 16)
            self.display_size[0] = self._align_up(self.display_size[0], 16)
            
            # 初始化显示
            Display.init(Display.ST7701, width=self.display_size[0], 
                        height=self.display_size[1], to_ide=True)
            
            # 初始化Sensor（先创建实例）
            self._sensor = Sensor()
            self._sensor.reset()
            
            # 【修复】使用实例属性获取像素格式
            # 配置通道0：给AI用 (RGB888)
            self._sensor.set_framesize(chn=CAM_CHN_ID_0, 
                                       width=self.rgb888p_size[0], 
                                       height=self.rgb888p_size[1])
            self._sensor.set_pixformat(self._sensor.RGB888, chn=CAM_CHN_ID_0)
            
            # 配置通道2：给显示用 (YUV420SP)
            self._sensor.set_framesize(chn=CAM_CHN_ID_2, 
                                       width=self.display_size[0], 
                                       height=self.display_size[1])
            self._sensor.set_pixformat(self._sensor.YUV420SP, chn=CAM_CHN_ID_2)
            
            # 初始化媒体缓冲区（在 sensor 配置后，run 之前）
            MediaManager.init()
            self._media_initialized = True
            
            # 绑定显示
            bind_info = self._sensor.bind_info(x=0, y=0, chn=CAM_CHN_ID_2)
            Display.bind_layer(**bind_info, layer=Display.LAYER_VIDEO1)
            
            # 创建OSD图层
            self._osd_img = image.Image(self.display_size[0], self.display_size[1], image.ARGB8888)
            
            # 启动Sensor
            self._sensor.run()
            
            # 设置帧率
            self.set_fps(fps)
            
            self._running = True
            self._paused = False
            
            print("[Camera] Started at %d fps" % fps)
            return True
            
        except Exception as e:
            print("[Camera] Start error:", e)
            import sys
            sys.print_exception(e)
            self.stop()
            return False
    
    def stop(self):
        """停止摄像头"""
        if not self._running and not self._media_initialized:
            return
        
        print("[Camera] Stopping...")
        
        try:
            if self._sensor:
                self._sensor.stop()
                self._sensor = None
            
            Display.deinit()
            
            if self._media_initialized:
                MediaManager.deinit()
                self._media_initialized = False
            
            self._osd_img = None
            self._running = False
            
            gc.collect()
            print("[Camera] Stopped")
            
        except Exception as e:
            print("[Camera] Stop error:", e)
    
    def get_frame(self):
        """
        获取当前帧（RGB888格式，用于AI处理）
        返回: Image对象 或 None
        """
        if not self._running or self._paused:
            return None
        
        # 帧率控制
        current_time = time.ticks_ms()
        elapsed = time.ticks_diff(current_time, self._last_frame_time)
        if elapsed < self._frame_interval_ms:
            time.sleep_ms(self._frame_interval_ms - elapsed)
        self._last_frame_time = time.ticks_ms()
        
        try:
            return self._sensor.snapshot(chn=CAM_CHN_ID_0)
        except Exception as e:
            print("[Camera] Get frame error:", e)
            return None
    
    def get_osd(self):
        """获取OSD图层用于绘制"""
        return self._osd_img
    
    def show(self):
        """刷新显示"""
        if self._osd_img and self._running:
            Display.show_image(self._osd_img, 0, 0, Display.LAYER_OSD3)
    
    def clear_osd(self):
        """清空OSD图层"""
        if self._osd_img:
            self._osd_img.clear()
    
    def set_fps(self, fps):
        """设置帧率 (5-30)"""
        if isinstance(fps, str):
            fps = self.FPS_LEVELS.get(fps, 30)
        
        fps = max(5, min(30, int(fps)))
        self._frame_interval_ms = 1000 // fps
        self._current_fps = fps
        print("[Camera] FPS set to:", fps)
    
    def get_fps(self):
        """获取当前帧率"""
        return self._current_fps
    
    def pause(self):
        """暂停取帧"""
        self._paused = True
        print("[Camera] Paused")
    
    def resume(self):
        """恢复取帧"""
        self._paused = False
        print("[Camera] Resumed")
    
    def is_running(self):
        """是否正在运行"""
        return self._running and not self._paused
    
    def get_rgb_size(self):
        """获取RGB帧尺寸"""
        return self.rgb888p_size.copy()
    
    def get_display_size(self):
        """获取显示尺寸"""
        return self.display_size.copy()
    
    def _align_up(self, value, align):
        return ((value + align - 1) // align) * align


def get_camera():
    """便捷函数：获取摄像头实例"""
    return CameraManager.get_instance()