# K230 AI Vision Controller

基于嘉楠 K230 的 AI 视觉从机控制器，支持人脸检测、人脸识别等功能。

> 🎯 **核心特点**：摄像头 24 小时常驻运行，AI 模型按需热切换，彻底解决资源阻塞问题。

## 📋 目录

- [功能特性](#功能特性)
- [硬件要求](#硬件要求)
- [快速开始](#快速开始)
- [命令协议](#命令协议)
- [架构设计](#架构设计)
- [开发历程](#开发历程)
- [常见问题](#常见问题)

## 功能特性

- ✅ **人脸检测** - 实时检测画面中的人脸位置
- ✅ **人脸识别** - 识别已注册的人脸身份
- ✅ **动态帧率** - 空闲时自动降帧，节省功耗
- ✅ **串口控制** - 通过 UART 接收命令，发送检测结果
- ✅ **热切换** - 毫秒级切换 AI 功能，无需重启摄像头

## 硬件要求

- 嘉楠 K230 开发板（CanMV-K230）
- ST7701 LCD 显示屏（或兼容屏幕）
- GC2093 摄像头模组
- UART 连接主控 MCU

## 快速开始

### 1. 文件部署

将以下文件拷贝到 K230 的 `/sdcard/` 目录：

```
/sdcard/
├── libs/
│   ├── K230Protocol.py
│   ├── K230SlaveController.py
│   ├── K230Uart.py
│   ├── CameraManager.py
│   ├── AIProcessor.py
│   ├── AIBase.py
│   └── AI2D.py
├── kmodel/
│   ├── face_detection_320.kmodel
│   └── face_recognition.kmodel
├── utils/
│   └── prior_data_320.bin
└── main.py
```

### 2. 运行程序

```python
# 在 K230 REPL 中执行
exec(open('/sdcard/main.py').read())
```

### 3. 发送命令测试

通过串口发送（波特率 115200）：

```
$CMD,PING#           → 返回 $RSP,PONG,K230#
$CMD,START,6#        → 启动人脸检测
$CMD,START,8#        → 启动人脸识别
$CMD,STOP#           → 停止当前功能
```

## 命令协议

### 命令格式

```
$CMD,<命令>[,参数1,参数2...]#
```

### 响应格式

```
$RSP,<长度>,<状态>,<消息>#
```

### 数据格式

```
$<长度>,<功能ID>,<x>,<y>,<w>,<h>[,额外数据]#
```

### 完整命令列表

| 命令 | 说明 | 示例 |
|------|------|------|
| `PING` | 测试连接 | `$CMD,PING#` |
| `STATUS` | 查询状态 | `$CMD,STATUS#` |
| `START,<id>` | 启动功能 | `$CMD,START,6#` |
| `STOP` | 停止功能 | `$CMD,STOP#` |
| `SET,fps,<value>` | 设置帧率 | `$CMD,SET,fps,15#` |
| `GET,fps` | 获取帧率 | `$CMD,GET,fps#` |
| `LIST` | 列出注册用户 | `$CMD,LIST#` |
| `DELETE,<id>` | 删除用户 | `$CMD,DELETE,zhangsan#` |
| `RELOAD` | 重载数据库 | `$CMD,RELOAD#` |
| `RESET` | 系统重置 | `$CMD,RESET#` |

### 功能 ID

| ID | 功能 |
|----|------|
| 6 | 人脸检测 |
| 8 | 人脸识别 |

## 架构设计

### 系统架构图

```
┌──────────────────────────────────────────────────────────────┐
│                        主控 MCU                               │
│                    (发送命令/接收结果)                         │
└─────────────────────────┬────────────────────────────────────┘
                          │ UART (115200bps)
┌─────────────────────────┴────────────────────────────────────┐
│                        K230 从机                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                   K230SlaveController                   │  │
│  │  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐  │  │
│  │  │  K230Uart   │  │ K230Protocol│  │   主循环       │  │  │
│  │  │  串口收发   │  │  协议解析   │  │ 命令+AI处理   │  │  │
│  │  └─────────────┘  └─────────────┘  └────────────────┘  │  │
│  └────────────────────────────────────────────────────────┘  │
│                              │                                │
│         ┌────────────────────┴────────────────────┐          │
│         ▼                                          ▼          │
│  ┌──────────────┐                          ┌──────────────┐  │
│  │CameraManager │                          │ AIProcessor  │  │
│  │ (24H 常驻)   │ ───── get_frame() ─────▶ │ (按需加载)   │  │
│  │              │                          │              │  │
│  │ • 摄像头控制 │                          │ • 模型管理   │  │
│  │ • 帧率调节   │                          │ • 推理处理   │  │
│  │ • OSD 绘制   │                          │ • 结果输出   │  │
│  └──────────────┘                          └──────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### 核心设计理念

**摄像头与 AI 解耦**

- `CameraManager`：负责摄像头的生命周期，启动后常驻运行
- `AIProcessor`：负责 AI 模型的加载/卸载/推理

这种设计实现了：
- ⚡ 功能切换从 **3 秒** 降到 **<100ms**
- 🛡️ 避免了媒体资源频繁开关导致的死锁
- 🔋 支持空闲时降帧以减少发热

## 开发历程

> 这是一段与 K230 媒体系统"斗智斗勇"的血泪史...

### 第一阶段：单线程方案（阻塞地狱）

**问题**：串口命令处理和 AI 推理都在主循环，命令响应严重延迟。

```python
# 原始方案：单线程
while True:
    cmd = uart.read()           # 阻塞！
    if cmd == "START":
        pipeline.create()        # 阻塞 2-3 秒！
        while running:
            frame = get_frame()  # AI 处理
            process(frame)
        pipeline.destroy()       # 又阻塞！
```

**症状**：发送 STOP 命令后，需要等当前帧处理完才能响应，体验极差。

---

### 第二阶段：多线程方案（死锁深渊）

**想法**：用 `_thread` 模块分离命令处理和 AI 推理。

```python
# 多线程方案
def ai_thread():
    pipeline.create()   # 💀 在子线程初始化媒体
    while not stop:
        process()
    pipeline.destroy()  # 💀 在子线程销毁媒体

_thread.start_new_thread(ai_thread, ())
```

**结果**：

```
[AI Thread] Initializing...
find sensor gc2093_csi2, type 8, output 1920x1080@30
vb common pool count 6
# 💀 卡死，永远不动了
```

**原因**：K230 的 `sensor` 和 `MediaManager` **只能在主线程操作**！在子线程初始化会导致底层驱动死锁。

---

### 第三阶段：混合模式（依然卡顿）

**尝试**：主线程负责初始化/销毁，子线程只做推理。

```python
# 混合方案
def main():
    pipeline.create()           # 主线程初始化
    _thread.start_new_thread(ai_loop, ())
    while True:
        cmd = uart.read()
        if cmd == "STOP":
            stop_flag = True
            pipeline.destroy()  # 主线程销毁
```

**问题**：每次切换功能仍需要完整的 `create()` → `destroy()` 周期，耗时 2-3 秒，而且偶尔还是会卡住。

---

### 第四阶段：灵光一现（终极方案）

> 💡 "既然开关摄像头这么麻烦，那就... **不关了**！"

**顿悟**：

```
摄像头开关 = 痛苦的根源
摄像头常驻 = 一劳永逸
```

**新架构**：

```python
# 终极方案：摄像头常驻
class CameraManager:
    def start(self):
        # 启动一次，永不关闭
        sensor.run()
    
    def get_frame(self):
        # 随时获取帧
        return sensor.snapshot()

class AIProcessor:
    def load(self, model_type):
        # 只加载 AI 模型，不碰摄像头
        self.model = load_kmodel(...)
    
    def unload(self):
        # 只卸载模型
        self.model.deinit()
```

**效果**：

| 操作 | 原方案 | 新方案 |
|------|--------|--------|
| 启动人脸检测 | 2.5s | 0.1s |
| 切换到人脸识别 | 3.0s | 0.15s |
| 停止 | 0.5s | 即时 |
| 稳定性 | 偶尔卡死 | 稳定 |

---

### 经验总结

1. **K230 媒体资源必须在主线程操作** - 这是铁律
2. **减少资源开关次数** - 能常驻就常驻
3. **分离变与不变** - 摄像头不变，AI 模型可变
4. **帧率动态调节** - 空闲时降帧，节省功耗

## 常见问题

### Q: 启动时报 `NameError: name 'xxx' isn't defined`

**A**: 检查导入语句，确保 `from media.xxx import *` 正确。

### Q: 画面卡住不动

**A**: 检查是否调用了 `camera.show()`，即使空闲状态也需要刷新显示。

### Q: 模型加载失败

**A**: 确认 `/sdcard/kmodel/` 目录下有对应的 `.kmodel` 文件。

### Q: 串口收不到数据

**A**: 
- 检查波特率是否为 115200
- 确认命令格式正确：`$CMD,PING#`（注意结尾的 `#`）

### Q: 切换功能时偶尔失败

**A**: 先发送 `$CMD,STOP#` 停止当前功能，再发送 `START`。

## 致谢

- 嘉楠科技 K230 SDK
- CanMV 开源社区
- Claude / Gemini / Chatgpt / Qwen AI 辅助开发

## License

MIT License

---

**Made with 💻 and ☕ by [Exmeaning]**

*"调试 K230 的日子，让我学会了什么叫'柳暗花明又一村'。"*
```
