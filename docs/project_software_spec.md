# 软件架构规格说明书 (Software Architecture Specification)

## 1. 架构概览

### 1.1 架构风格
本系统采用 **分层架构 (Layered Architecture)** 与 **管道-过滤器 (Pipeline-Filter)** 模式相结合的设计。
- **分层**: 将硬件抽象、算法处理、业务逻辑与用户界面分离，降低耦合度。
- **管道**: 传感器数据流经一系列处理节点（采集 -> 对齐 -> 追踪 -> 融合 -> 渲染/存储），各节点通过线程安全的队列解耦。

### 1.2 技术栈
- **编程语言**: Python 3.8+
- **GUI 框架**: PyQt5 / PySide6 或 DearPyGui (用于高性能实时绘图)
- **计算机视觉**: OpenCV, NumPy
- **人体姿态估计**: MediaPipe (Pose)
- Kinect: `pyk4a` (Azure Kinect)
  - IMU: `pyserial`
- **数学计算**: SciPy (四元数/旋转矩阵)

## 2. 系统层次结构

系统自下而上分为四层：硬件抽象层、算法核心层、业务逻辑层、应用展示层。

```mermaid
graph TD
    subgraph "Hardware Abstraction Layer"
        HAL_Kinect[Azure Kinect Driver]
        HAL_IMU[IMU Driver]
    end

    subgraph "Algorithm Core Layer"
        Algo_Pre[Preprocessing\n(Undistort, Align)]
        Algo_Body[Body Tracker\n(MediaPipe Pose)]
        Algo_Brush[Brush Tracker\n(Vision+IMU)]
        Algo_Fusion[Sensor Fusion]
    end

    subgraph "Application Logic Layer"
        Logic_Sync[Time Synchronizer]
        Logic_Rec[Recorder]
        Logic_Exp[CSV Exporter]
        Logic_Config[Config Manager]
    end

    subgraph "Presentation Layer"
        UI_Main[Main Window]
        UI_Render[Overlay Renderer]
        UI_Control[Control Panel]
    end

    HAL_Kinect --> Algo_Pre
    Algo_Pre --> Algo_Body
    Algo_Pre --> Algo_Brush
    HAL_IMU --> Algo_Fusion
    Algo_Brush --> Algo_Fusion
    
    Algo_Body --> Logic_Sync
    Algo_Fusion --> Logic_Sync
    
    Logic_Sync --> UI_Render
    Logic_Sync --> Logic_Rec
```

## 3. 模块详细设计

### 3.1 硬件抽象层 (HAL)
- **KinectCamera**: 
  - 封装 `pyk4a` 调用，固定 720p / 5fps / NV12 / NFOV_UNBINNED。
  - 提供 `read_frame()` 接口，返回包含 RGB(BGR)、Depth、Infrared 及内参的 `Frame`。
  - NV12 转换为 BGR 后提供给上层。
- **WitMotionSensor**:
  - 管理串口连接与数据包解析。
  - 维护一个基于时间戳的 Ring Buffer，供融合算法按时间查询 IMU 数据。

### 3.1.1 I/O 与数据集
- **FileCamera**:
  - 从目录结构读取录制的彩色与深度图，解析 `meta.json`，输出 `Frame`。
  - 支持可选 IR 图像。
- **KinectRawRecorder**:
  - 采集并将彩色图 (NV12 转 PNG)、深度图 (16-bit PNG)、`meta.json` 写入指定目录。
  - 提供 PowerShell 包装脚本以 `uv` 方式运行。

### 3.2 算法核心层 (Algorithm Core)
- **BodyTracker**:
  - 封装 MediaPipe Pose 管线。
  - 输入: RGB 图像。
  - 输出: `Skeleton` 对象 (关键点列表)。
  - 职责: 使用 MediaPipe Pose 估计 2D 关键点，并结合 Kinect 深度图做坐标提升 (2D->3D)。
- **BrushTracker**:
  - 输入: RGB 图像 + 深度图。
  - 输出: 视觉计算得到的 `BrushPoseVis` (Pos, Dir)。
  - 职责: 执行 HSV 颜色分割、连通域分析、反投影。
- **PoseFuser**:
  - 输入: `BrushPoseVis` + IMU 数据片段。
  - 输出: 最终 `BrushPose` (Pos, Quat)。
  - 职责: 执行互补滤波，处理时间对齐，计算最终四元数。

### 3.3 业务逻辑层 (Application Logic)
- **TimeSynchronizer**:
  - 系统的“心跳”协调者。
  - 接收各路算法结果，按 Frame ID 或 Timestamp 对齐打包成 `FrameResult`。
- **Recorder**:
  - 接收 `FrameResult` 和原始图像。
  - 视频流写入: 使用 `cv2.VideoWriter` (多线程)。
  - 数据流缓存: 内存缓冲 -> 异步写入磁盘。
- **ConfigManager**:
  - 单例模式，加载/保存 `config.json`。
  - 管理相机内参、HSV 阈值、IMU 标定矩阵。

### 3.4 应用展示层 (Presentation)
- **MainWindow**: 主窗口容器。
- **Visualizer**: 
  - 基于 OpenCV 或 OpenGL 的画布。
  - 负责将 RGB 图像与骨架/笔杆的 3D 投影叠加绘制。
- **ControlPanel**:
  - 阈值调节滑块 (Signal-Slot 机制实时更新 Config)。
  - 录制/停止按钮。

## 4. 数据流与并发模型

### 4.1 线程模型
系统至少包含以下独立线程：
1. **Main GUI Thread**: 处理 UI 事件、渲染最终画面。
2. **Kinect Capture Thread**: 循环读取相机帧 (30FPS)。
3. **IMU Capture Thread**: 循环读取串口数据 (100Hz+)。
4. **Processing Worker Thread**: 
   - 从采集队列取数据 -> 运行算法 -> 放入渲染队列。
   - 为避免 UI 卡顿，重型算法 (YOLO11 Pose) 必须在此线程运行。
5. **IO Thread**: 负责视频编码与文件写入，避免磁盘 IO 阻塞处理流程。

### 4.2 数据结构
- **Frame**: `{ color, depth, ir, timestamp, frame_id }`
- **Skeleton**: `{ joints: List[Point3D], confidence: float }`
- **BrushPose**: `{ position: Vec3, rotation: Quaternion, is_tracked: bool }`
- **FrameResult**: `{ frame_id, timestamp, skeleton, brush_pose }`

## 5. 接口设计

### 5.1 插件化接口
为了方便算法升级，`Tracker` 应定义基类接口：

```python
class IBrushTracker(ABC):
    @abstractmethod
    def track(self, color_img, depth_img) -> BrushPoseVis:
        pass
```

### 5.2 消息传递
各层之间通过 `Queue` 进行通信：
- `raw_frame_queue`: HAL -> Processing
- `result_queue`: Processing -> GUI / Recorder

### 5.3 原始数据集目录结构
- `color/000001.png ...`
- `depth/000001.png ...` (uint16, mm)
- `ir/000001.png ...` 可选
- `meta.json`:
  - `intrinsics`: `{fx, fy, cx, cy}`
  - `fps`: `5.0`
  - `timestamps`: 数组
