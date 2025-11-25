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
    
    使用方法：
        cam = CameraManager.get_instance()
        cam.start()
        
        while True:
            frame = cam.get_frame()
            # 处理帧...
            cam.get_osd().draw_rectangle(...)
            cam.show()
    """
    
    _instance = None
    
    # 支持的帧率档位
    FPS_LEVELS = {
        'high': 30,
        'medium': 15,
        'low': 10,
        'idle': 5
    }
    
    @classmethod
    def get_instance(cls):
        """获取单例实例"""
        if cls._instance is None:
            cls._instance = CameraManager()
        return cls._instance
    
    def __init__(self):
        if CameraManager._instance is not None:
            raise Exception("Use CameraManager.get_instance()")
        
        # 配置参数
        self.rgb888p_size = [640, 480]
        self.display_size = [640, 480]
        self.sensor_id = 2       # CSI2
        self.sensor_chn = CAM_CHN_ID_0
        self.display_chn = DISPLAY_CHN_VIDEO1
        
        # 状态
        self._running = False
        self._paused = False
        self._current_fps = 30
        self._target_fps = 30
        
        # 媒体对象
        self._sensor = None
        self._display = None
        self._osd_img = None
        self._media_initialized = False
        
        # 帧率控制
        self._frame_interval = 1.0 / 30  # 默认30fps
        self._last_frame_time = 0
        self._skip_frames = False
        
        print("[Camera] Manager created")
    
    def start(self, fps=30):
        """启动摄像头"""
        if self._running:
            print("[Camera] Already running")
            return True
        
        print("[Camera] Starting...")
        
        try:
            # 初始化显示
            display_type = Display.ST7701  # 根据实际屏幕修改
            Display.init(display_type, width=self.display_size[0], 
                        height=self.display_size[1], to_ide=True)
            
            # 初始化媒体缓冲区
            config = k_vb_config()
            config.max_pool_cnt = 1
            config.comm_pool[0].blk_size = 4 * self.display_size[0] * self.display_size[1]
            config.comm_pool[0].blk_cnt = 1
            config.comm_pool[0].mode = VB_REMAP_MODE_NOCACHE
            
            ret = MediaManager.init(config)
            if ret:
                raise Exception("MediaManager init failed: %d" % ret)
            
            self._media_initialized = True
            
            # 创建OSD图层
            self._osd_img = image.Image(self.display_size[0], self.display_size[1], image.ARGB8888)
            Display.bind_layer(**{
                'channel': self.display_chn,
                'layer': Display.LAYER_OSD3,
                'rect': (0, 0, self.display_size[0], self.display_size[1]),
                'pix_format': PIXEL_FORMAT_ARGB_8888,
                'flag': Display.LAYER_OSD_SYNC
            })
            
            # 初始化Sensor
            self._sensor = Sensor(id=self.sensor_id)
            self._sensor.reset()
            
            # 配置通道0：给AI用 (RGB888P)
            self._sensor.set_framesize(
                w=self.rgb888p_size[0],
                h=self.rgb888p_size[1],
                chn=CAM_CHN_ID_0
            )
            self._sensor.set_pixformat(Sensor.RGB888P, chn=CAM_CHN_ID_0)
            
            # 配置通道2：给显示用 (YUV420SP)
            self._sensor.set_framesize(
                w=self.display_size[0],
                h=self.display_size[1],
                chn=CAM_CHN_ID_2
            )
            self._sensor.set_pixformat(Sensor.YUV420SP, chn=CAM_CHN_ID_2)
            
            # 绑定显示
            sensor_bind_info = self._sensor.bind_info(
                x=0, y=0,
                chn=CAM_CHN_ID_2
            )
            Display.bind_layer(**sensor_bind_info, layer=Display.LAYER_VIDEO1)
            
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
        获取当前帧（RGB888P格式，用于AI处理）
        返回: numpy数组 或 None
        """
        if not self._running or self._paused:
            return None
        
        # 帧率控制
        if self._skip_frames:
            current_time = time.ticks_ms() / 1000.0
            elapsed = current_time - self._last_frame_time
            if elapsed < self._frame_interval:
                time.sleep_ms(int((self._frame_interval - elapsed) * 1000))
            self._last_frame_time = time.ticks_ms() / 1000.0
        
        try:
            img = self._sensor.snapshot(chn=CAM_CHN_ID_0)
            return img.to_numpy_ref() if img else None
        except Exception as e:
            print("[Camera] Get frame error:", e)
            return None
    
    def get_osd(self):
        """获取OSD图层用于绘制"""
        return self._osd_img
    
    def show(self):
        """刷新显示"""
        if self._osd_img and self._running:
            Display.show_image(self._osd_img, 0, 0, self.display_chn, Display.LAYER_OSD3)
    
    def clear_osd(self):
        """清空OSD图层"""
        if self._osd_img:
            self._osd_img.clear()
    
    def set_fps(self, fps):
        """
        设置帧率
        
        参数:
            fps: 帧率值 (5-30) 或 预设名称 ('high', 'medium', 'low', 'idle')
        """
        # 处理预设名称
        if isinstance(fps, str):
            fps = self.FPS_LEVELS.get(fps, 30)
        
        # 限制范围
        fps = max(5, min(30, int(fps)))
        
        if fps == self._current_fps:
            return
        
        self._target_fps = fps
        self._frame_interval = 1.0 / fps
        
        # 如果目标帧率低于30，使用软件跳帧
        if fps < 30:
            self._skip_frames = True
        else:
            self._skip_frames = False
        
        self._current_fps = fps
        print("[Camera] FPS set to:", fps)
    
    def get_fps(self):
        """获取当前帧率"""
        return self._current_fps
    
    def pause(self):
        """暂停取帧（摄像头仍运行，只是不处理）"""
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


# ========== 便捷函数 ==========

def get_camera():
    """获取摄像头实例"""
    return CameraManager.get_instance()