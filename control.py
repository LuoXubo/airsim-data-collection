import setup_path 
import airsim
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入三维绘图模块
import cv2
import os

save_path = 'data\\height75fov90'
image_save_path = os.path.join(save_path, 'images')
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)
save_str = ''

# 初始化 AirSim 客户端
client = airsim.VehicleClient()
client.confirmConnection()

# 获取初始位置和姿态
pose = client.simGetVehiclePose()
print("初始位置: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
time.sleep(2)
timestamp = time.time()

# 用于记录轨迹的列表
xs, ys, zs = [], [], []

num_stage1 = 400
num_stage2 = 100
num_stage3 = 100

stage1_2 = 300
stage2_3 = 400

# 第一阶段：巡视阶段
print("开始第一阶段：巡视阶段")
stage1_v = np.linspace(0, stage1_2, num_stage1)  # x 从 0 到 100
stage1_o = np.linspace(-60, -60, num_stage1)  # pitch 保持 -60°

idx = 0

for i in range(num_stage1):
    v = stage1_v[i]
    o = stage1_o[i]
    
    # 位置更新
    pose.position.x_val = v
    # pose.position.z_val = (v / 100) * 10  # z 从 0 到 10，线性下降（慢）
    pose.position.z_val = 0  # 保持 z=0
    pose.position.y_val = 0  # 保持 y=0
    
    # 姿态更新
    # pitch = -45 * (x / 100) * math.pi / 180  # pitch 从 0° 到 -45°，线性变化
    pitch = o * math.pi / 180
    roll = 0
    yaw = 0
    pose.orientation = airsim.to_quaternion(pitch, roll, yaw)
    
    # 记录轨迹
    xs.append(pose.position.x_val)
    ys.append(pose.position.y_val)
    zs.append(pose.position.z_val)
    
    # 设置无人机位置和姿态
    client.simSetVehiclePose(pose, True)
    print("巡视阶段: x={}, y={}, z={}, pitch={}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val, o))
    
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    ])
    rgb_response = responses[0]
    rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    if rgb_img1d.size > 0:
        rgb_img_rgb = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)
        image_idx = str(idx).zfill(4)
        cv2.imwrite(os.path.join(image_save_path, f'image_{image_idx}.png'), rgb_img_rgb)
        save_str += f'{timestamp} images/image_{image_idx}.png {pose.position.x_val} {pose.position.y_val} {pose.position.z_val} {pose.orientation.w_val} {pose.orientation.x_val} {pose.orientation.y_val} {pose.orientation.z_val}\n'
        idx += 1
        timestamp += 0.04
    else:
        print("未获取到 RGB 图像数据")

print("巡视阶段结束")

# 第二阶段: 转向阶段
print("开始第二阶段：转向阶段")
stage2_v = np.linspace(stage1_2, stage2_3, num_stage2)  # x 从 100 到 150，水平移动减慢
stage2_o = np.linspace(-60, -90, num_stage2)  # pitch 从 -15° 到 -90°，快速下降

for i in range(num_stage2):
    v = stage2_v[i]
    o = stage2_o[i]
    
    # 位置更新
    t = (v - stage1_2) / num_stage2  # t 从 0 到 1，规范化参数
    pose.position.x_val = v
    pose.position.z_val = (t ** 2) * 10
    pose.position.y_val = 0
    
    pitch_rad = o * math.pi / 180  # 转换为弧度
    roll = 0
    yaw = 0
    pose.orientation = airsim.to_quaternion(pitch_rad, roll, yaw)
    
    # 记录轨迹
    xs.append(pose.position.x_val)
    ys.append(pose.position.y_val)
    zs.append(pose.position.z_val)
    
    # 设置无人机位置和姿态
    client.simSetVehiclePose(pose, True)
    print("转向阶段: x={}, y={}, z={}, pitch={}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val, o))
    
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    ])
    rgb_response = responses[0]
    rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    if rgb_img1d.size > 0:
        rgb_img_rgb = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)
        image_idx = str(idx).zfill(4)
        cv2.imwrite(os.path.join(image_save_path, f'image_{image_idx}.png'), rgb_img_rgb)
        save_str += f'{timestamp} images/image_{image_idx}.png {pose.position.x_val} {pose.position.y_val} {pose.position.z_val} {pose.orientation.w_val} {pose.orientation.x_val} {pose.orientation.y_val} {pose.orientation.z_val}\n'
        idx += 1
        timestamp += 0.04
    else:
        print("未获取到 RGB 图像数据")

print("转向阶段结束")

# 第三阶段：降落阶段
print("开始第三阶段：降落阶段")
stage3_v = np.linspace(stage2_3, stage2_3, num_stage3)  # x 保持 125
stage3_o = np.linspace(-90, -90, num_stage3)  # pitch 保持 -90°

for i in range(num_stage3):
    v = stage3_v[i]
    o = stage3_o[i]
    
    # 位置更新
    pose.position.x_val = v
    pose.position.z_val += 0.1  # z 从 10 缓慢增加
    pose.position.y_val = 0
    
    pitch_rad = o * math.pi / 180  # 转换为弧度
    roll = 0
    yaw = 0
    pose.orientation = airsim.to_quaternion(pitch_rad, roll, yaw)
    
    # 记录轨迹
    xs.append(pose.position.x_val)
    ys.append(pose.position.y_val)
    zs.append(pose.position.z_val)
    
    # 设置无人机位置和姿态
    client.simSetVehiclePose(pose, True)
    print("降落阶段: x={}, y={}, z={}, pitch={}".format(
        pose.position.x_val, pose.position.y_val, pose.position.z_val, o))
    
    responses = client.simGetImages([
        airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False),
    ])
    rgb_response = responses[0]
    rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
    if rgb_img1d.size > 0:
        rgb_img_rgb = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)
        image_idx = str(idx).zfill(4)
        cv2.imwrite(os.path.join(image_save_path, f'image_{image_idx}.png'), rgb_img_rgb)
        save_str += f'{timestamp} images/image_{image_idx}.png {pose.position.x_val} {pose.position.y_val} {pose.position.z_val} {pose.orientation.w_val} {pose.orientation.x_val} {pose.orientation.y_val} {pose.orientation.z_val}\n'
        idx += 1
        timestamp += 0.04
    else:
        print("未获取到 RGB 图像数据")

with open(os.path.join(save_path, 'pose.txt'), 'w') as f:
    f.write(save_str)
    f.close()

# 绘制三维轨迹
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xs, ys, zs)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()