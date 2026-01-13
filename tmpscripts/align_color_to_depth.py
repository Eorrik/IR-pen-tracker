import pyrealsense2 as rs
import numpy as np
import cv2

# 1. 配置 Pipeline
pipeline = rs.pipeline()
config = rs.config()

# 启用流：Color, Depth, Infrared (Left)
# 注意：深度流必须开启，因为对齐算法依赖深度数据来计算坐标映射
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30) # 1 表示 Left IR

# 2. 创建对齐对象
# RS2_STREAM_DEPTH 意味着我们要把其他所有流（这里是Color）对齐到 Depth（即IR视角）
align_to = rs.stream.depth 
align = rs.align(align_to)

# 3. 启动
profile = pipeline.start(config)

try:
    while True:
        # 获取帧集
        frames = pipeline.wait_for_frames()
        
        # 执行对齐：这将生成一个新的帧集，其中 color frame 已经被 warp 到了 depth 视角
        aligned_frames = align.process(frames)
        
        # 获取对齐后的帧
        aligned_depth_frame = aligned_frames.get_depth_frame() # 原始视角
        aligned_ir_frame = aligned_frames.get_infrared_frame() # 原始视角 (无需变化)
        aligned_color_frame = aligned_frames.get_color_frame() # **这是被变形后的 Color**

        # 验证是否获取成功
        if not aligned_depth_frame or not aligned_color_frame or not aligned_ir_frame:
            continue

        # 转换为 Numpy 数组
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())
        ir_image = np.asanyarray(aligned_ir_frame.get_data())

        # 此时：color_image 和 ir_image 像素是完全对齐的
        # 可以用 cv2.addWeighted 叠加显示
        ir_3ch = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2BGR) # 转为3通道方便叠加
        overlay = cv2.addWeighted(color_image, 0.5, ir_3ch, 0.5, 0)

        cv2.imshow('Aligned Color to IR', overlay)
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()