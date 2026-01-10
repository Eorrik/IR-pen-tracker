# 核心算法规范 (Core Algorithm Specification)

本文档详细说明本系统的核心算法实现原理，包括毛笔 6-DoF 姿态追踪（视觉+IMU 融合）与人体姿态追踪（基于 MediaPipe Pose + RGB Lifting）。

## 0. 录制与回放 (Raw Recording & Playback)
- 为支持离线调试与回放，系统提供原始数据录制与读取能力：
  - 录制: 使用 KinectCamera 采集，NV12 转 BGR，深度写 16-bit PNG，并生成 `meta.json`。
  - 回放: 使用 FileCamera 从目录读取 `color/depth/meta.json`，输出统一 `Frame` 供算法管线消费。

## 1. 毛笔姿态追踪算法 (Brush Pose Tracking)

毛笔姿态追踪采用“视觉定位 + IMU 姿态融合”的混合架构。视觉部分负责绝对位置与漂移校正，IMU 负责高频姿态更新与 Roll 轴解算。

### 1.1 视觉追踪 (Vision Tracking)

视觉追踪的目标是从 Kinect 的图像流中提取笔杆上的两个特征点（Marker A 和 Marker B），并计算其 3D 坐标。

#### 1.1.1 特征提取
- **基于 RGB 的颜色分割**: 使用 HSV 颜色空间分割。
  - 将 RGB 图像转换至 HSV 空间。
  - 应用预设的颜色阈值 (Masking) 提取红色（笔锋端）和蓝色（笔尾端）区域。
  - 对二值 Mask 进行形态学操作（开运算）去除噪点。
  - 计算最大连通域的质心 $(u_r, v_r)$ 和 $(u_b, v_b)$。

#### 1.1.2 3D 反投影与圆柱拟合 (3D Back-Projection & Cylinder Fitting)

由于使用了高反射率的 IR 贴纸，IR 图像中的标记点非常清晰（亮度 > 60000），因此 2D 像素坐标 $(u, v)$ 非常可靠。
然而，高强度的红外反射会导致 Time-of-Flight (ToF) 深度相机产生多径效应或饱和，导致标记点处的深度值 $Z$ **极不可靠**（通常表现为飞点或零值）。

因此，本系统采用 **"笔身深度拟合"** 策略：
1.  **利用 2D 标记点定位笔身范围**: 连接两个高亮 IR 标记点的 2D 连线，覆盖了笔身区域。
2.  **提取笔身深度**: 在两个标记点之间的连线区域采样深度数据。笔身通常为漫反射材质，其深度数据比高反标记点更准确。
3.  **圆柱/直线拟合**: 将采样的笔身 3D 点云拟合为一条 3D 直线（圆柱轴线）。
4.  **重构标记点 3D 坐标**: 将 IR 图像中检测到的高精度 2D 标记点 $(u, v)$，通过相机光心投射射线，与拟合出的 3D 轴线相交，交点即为修正后的标记点 3D 坐标。

此方法利用了笔的几何约束（刚体圆柱），规避了高反区域的深度噪声。

#### 1.1.3 视觉姿态解算
- **位置 (Position)**: 取拟合修正后的笔锋端 3D 坐标 $P_{tip}'$。
- **方向向量 (Direction)**: $\vec{V} = \text{normalize}(P_{blue}' - P_{red}')$，其中 $P'$ 为投影到拟合轴线上的点。
- **Pitch/Yaw**:
  - $Yaw = \arctan2(V_x, V_z)$
  - $Pitch = \arctan2(-V_y, \sqrt{V_x^2 + V_z^2})$
- **约束检查**: 计算 $\|P_{blue} - P_{red}\|$，若与物理笔杆标记间距偏差超过阈值（如 20%），则判定为 **Lost**。

### 1.2 IMU 数据处理
IMU 安装于笔杆末端，提供高频 (100Hz+) 的角速度和加速度数据。
- **坐标系对齐**:
  - IMU 坐标系与笔杆坐标系通常不重合。需通过**手眼标定**计算旋转矩阵 $R_{align}$。
  - 校准后的 IMU 四元数: $q_{calib} = R_{align} \otimes q_{raw}$。
- **Roll 轴获取**: 视觉无法检测绕笔杆轴线的自转。该自由度完全依赖 IMU 解算。

### 1.3 传感器融合 (Sensor Fusion)
采用 **互补滤波 (Complementary Filter)** 策略融合低频视觉数据 (30Hz) 与高频 IMU 数据。

**算法流程**:
1. **时间同步**: 以 Kinect 图像帧的时间戳 $t_{img}$ 为基准。寻找 IMU 缓冲区中时间戳最接近 $t_{img}$ 的数据包。
2. **姿态融合**:
   - **Yaw / Pitch**:
     $$ \theta_{fused} = \alpha \cdot \theta_{gyro} + (1 - \alpha) \cdot \theta_{vision} $$
     其中 $\alpha$ 为置信系数（如 0.98），$\theta_{gyro}$ 为上一帧融合结果叠加陀螺仪积分增量。
   - **Roll**:
     $$ \phi_{fused} = \phi_{imu} - \phi_{bias} $$
     直接使用 IMU 的 Roll 值（扣除初始零偏）。
3. **输出**: 合成最终四元数 $Q_{final}$。

---

## 2. 人体姿态追踪算法 (Human Pose Tracking) [Shelved]

> **Note**: This feature is currently shelved to focus on core pen tracking and app functionality.

人体姿态追踪采用 **MediaPipe Pose** 进行 2D 关键点检测（Landmarks），配合 Kinect 深度图进行 **RGB Lifting** 以获取 3D 坐标。

### 2.1 算法选型
- **模型**: **MediaPipe Pose**。
- **优势**:
  - **工程集成简单**: 直接输出人体关键点（Landmarks）及可见度/置信度。
  - **实时性好**: 适合 CPU 实时运行与交互式预览。
  - **平滑能力**: 可使用其内置时序平滑减少抖动。

### 2.2 实现管线 (Pipeline)

#### 2.2.1 模型推理 (Inference)
- **输入**: Kinect 获取的 Color 图像 (BGR -> RGB)。
- **引擎**: 使用 **MediaPipe Pose** 运行姿态估计。
- **输出**: 人体关键点集合 $N$，每个关键点提供归一化坐标 $(x_i, y_i)$、可见度/置信度 $c_i$。
  - 像素坐标换算: $u_i = x_i \cdot W,\; v_i = y_i \cdot H$。

#### 2.2.2 RGB Lifting (2D to 3D)
利用 Kinect 的 RGB-D 特性，将 2D 关键点提升为 3D 空间坐标。

1. **深度对齐**: 确保深度图已对齐至 RGB 镜头视角 (MapDepthToColor)。
2. **深度采样**:
   - 对每个关键点 $(u_i, v_i)$，在深度图 $D$ 中读取深度值 $Z_{raw}$。
   - **邻域滤波**: 由于单点深度可能存在噪声或无效值 (0)，在 $(u_i, v_i)$ 周围 $3\times3$ 或 $5\times5$ 区域内取有效深度的中值作为 $Z_i$。
3. **坐标反投影**:
   利用相机内参 $K$，将 $(u_i, v_i, Z_i)$ 转换为相机坐标系下的 3D 坐标：
   $$
   \begin{cases}
   z = Z_i \\
   x = \frac{(u_i - c_x) \cdot z}{f_x} \\
   y = \frac{(v_i - c_y) \cdot z}{f_y}
   \end{cases}
   $$
4. **平滑滤波**: 对生成的 3D 关键点序列应用 **OneEuroFilter** 或 **Kalman Filter** 以减少抖动。

### 2.3 性能优化
- **推理加速**: 使用 MediaPipe 的轻量管线并启用合理的输入分辨率与帧率。
- **异步处理**: 图像采集线程与姿态估计线程解耦，保证渲染或主逻辑不被阻塞。
