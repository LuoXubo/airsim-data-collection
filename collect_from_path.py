import setup_path 
import airsim
import math
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 导入三维绘图模块
import cv2
import os

traj_dir = 'data\\j_curve_trajectories'
save_dir = 'data\\collected_from_path'

traj_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.txt')])


client = airsim.VehicleClient()
client.confirmConnection()

initial_pose = client.simGetVehiclePose()
pose = initial_pose
print("初始位置: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
time.sleep(2)
# timestamp = time.time()

for traj_file in traj_files:
    pose = initial_pose
    print(f"Processing trajectory file: {traj_file}")
    
    save_name = f"collected_{traj_file}".split('.')[0]
    image_save_path = os.path.join(save_dir, save_name, 'images')
    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)

    groundtruth_str = ''
    list_str = ''
    
    data = np.loadtxt(os.path.join(traj_dir, traj_file), delimiter=',')
    xs, ys, zs = data[:, 0], data[:, 1], data[:, 2]
    rolls, pitches, yaws = data[:, 3], data[:, 4], data[:, 5]

    num_points = len(xs)
    for i in range(num_points):
        pose.position.x_val = xs[i]
        pose.position.y_val = ys[i]
        pose.position.z_val = zs[i]
        
        pitch = pitches[i]
        roll = rolls[i]
        yaw = yaws[i]
        
        pose.orientation = airsim.to_quaternion(pitch, roll, yaw)

        client.simSetVehiclePose(pose, ignore_collision=True)
        
        timestamp = time.time()
        image_name = f'camera_{i:04d}.png'
        
        groundtruth_str += f'{timestamp:.6f} {pose.position.x_val} {pose.position.y_val} {pose.position.z_val} {pose.orientation.w_val} {pose.orientation.x_val} {pose.orientation.y_val} {pose.orientation.z_val}\n'
        list_str += f'images/{image_name} {timestamp:.6f}\n'
        
        responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
        rgb_response = responses[0]
        rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        if rgb_img1d.size > 0:
            rgb_img_rgb = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)
            cv2.imwrite(os.path.join(image_save_path, image_name), rgb_img_rgb)
        else:
            print("未获取到 RGB 图像数据")
            
        time.sleep(0.05)
        
        
    # 保存 groundtruth 和 list 文件
    groundtruth_file = os.path.join(save_dir, save_name, 'groundtruth.txt')
    list_file = os.path.join(save_dir, save_name, 'images.txt')

    with open(groundtruth_file, 'w') as f:
        f.write(groundtruth_str)

    with open(list_file, 'w') as f:
        f.write(list_str)
        
print('All trajectories processed and data collected successfully.')