import setup_path 
import airsim
import math
import numpy as np
import time
import cv2
import os

traj_dir = 'data\\trajectories'
save_dir = 'data\\captured_images_trajectories_txt'

traj_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.txt')])


client = airsim.VehicleClient()
client.confirmConnection()

initial_pose = client.simGetVehiclePose()
pose = initial_pose
print("初始位置: x={}, y={}, z={}".format(pose.position.x_val, pose.position.y_val, pose.position.z_val))
time.sleep(2)

for traj_idx, traj_file in enumerate(traj_files):
    pose = initial_pose
    print(f"Processing trajectory file: {traj_file}")
    
    # 使用 trajectory_1, trajectory_2, ... 格式命名
    save_name = f"trajectory_{traj_idx + 1}"
    traj_save_path = os.path.join(save_dir, save_name)
    if not os.path.exists(traj_save_path):
        os.makedirs(traj_save_path)

    # 用于存储轨迹数据的列表
    positions = []
    orientations = []
    
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
        
        # 记录位置和姿态
        positions.append([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        orientations.append([pose.orientation.w_val, pose.orientation.x_val, 
                            pose.orientation.y_val, pose.orientation.z_val])
        
        # 图像命名为 0.jpg, 1.jpg, 2.jpg, ...
        image_name = f'{i}.jpg'
        
        responses = client.simGetImages([airsim.ImageRequest("front_center", airsim.ImageType.Scene, False, False)])
        rgb_response = responses[0]
        rgb_img1d = np.frombuffer(rgb_response.image_data_uint8, dtype=np.uint8)
        if rgb_img1d.size > 0:
            rgb_img_rgb = rgb_img1d.reshape(rgb_response.height, rgb_response.width, 3)
            cv2.imwrite(os.path.join(traj_save_path, image_name), rgb_img_rgb)
        else:
            print("未获取到 RGB 图像数据")
            
        time.sleep(0.05)
        
    # 保存 traj_data.txt 文件
    traj_data = np.column_stack([positions, orientations])  # shape: (T, 7), [x, y, z, qw, qx, qy, qz]
    
    txt_file = os.path.join(traj_save_path, 'traj_data.txt')
    np.savetxt(txt_file, traj_data, fmt='%.6f', delimiter=',')
    
    print(f"Saved {num_points} images and traj_data.txt to {traj_save_path}")
        
print('All trajectories processed and data collected successfully.')