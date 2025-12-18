import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class LunarLandingDataset(Dataset):
    def __init__(self, 
                 data_root, 
                 image_root, 
                 horizon=64, 
                 dt=0.1, 
                 include_velocity=True):
        """
        参数:
            data_root (str): 轨迹 .txt 文件的目录
            image_root (str): 图像序列的目录 (假设结构为 image_root/traj_XX/frame_YYY.png)
            horizon (int): Diffusion模型预测的未来时间步长 (例如 64 或 128)
            dt (float): 采样时间间隔，用于计算速度
            include_velocity (bool): 是否在状态中包含速度
        """
        self.data_root = data_root
        self.image_root = image_root
        self.horizon = horizon
        self.dt = dt
        self.include_velocity = include_velocity
        
        # 1. 加载所有轨迹文件
        self.traj_files = sorted(glob.glob(os.path.join(data_root, "*.txt")))
        self.trajectories = []
        
        print(f"Loading {len(self.traj_files)} trajectories...")
        
        # 2. 预处理：读取数据并计算速度
        for fpath in self.traj_files:
            # 数据格式: x, y, z, roll, pitch, yaw
            raw_data = np.loadtxt(fpath, delimiter=',')
            
            # 计算速度 (v = (p_next - p_curr) / dt)
            # 补齐最后一个点的速度为0
            pos = raw_data[:, :3]
            vel = np.zeros_like(pos)
            vel[:-1] = (pos[1:] - pos[:-1]) / self.dt
            vel[-1] = 0 # 假设着陆时速度为0
            
            # 拼接状态: [x, y, z, vx, vy, vz, roll, pitch, yaw]
            # 维度: (T, 9)
            if self.include_velocity:
                state = np.hstack([pos, vel, raw_data[:, 3:]])
            else:
                state = raw_data
                
            self.trajectories.append(state)

        # 3. 计算归一化统计量 (Statistics)
        # 我们需要基于"相对坐标"来计算均值方差，否则没有意义
        all_relative_states = []
        for traj in self.trajectories:
            # 转换为相对于终点的坐标 (Target-Centric)
            target = traj[-1, :3] # 取最后一个点的位置作为目标
            rel_traj = traj.copy()
            rel_traj[:, :3] -= target # 位置变为相对值
            all_relative_states.append(rel_traj)
            
        all_data = np.vstack(all_relative_states)
        self.stats = {
            'min': np.min(all_data, axis=0),
            'max': np.max(all_data, axis=0),
            'mean': np.mean(all_data, axis=0),
            'std': np.std(all_data, axis=0) + 1e-6 # 防止除零
        }
        
        # 图像预处理
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)), # 根据你的网络输入调整
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def normalize(self, data):
        """ 将真实物理值映射到标准正态分布 """
        return (data - self.stats['mean']) / self.stats['std']

    def unnormalize(self, data):
        """ 将模型输出还原为物理值 """
        return data * self.stats['std'] + self.stats['mean']

    def __len__(self):
        # 每一个时间步都可以作为一个训练样本
        # 这里的长度是所有轨迹的总帧数
        return sum([len(t) for t in self.trajectories])

    def __getitem__(self, idx):
        # 1. 找到 idx 对应的 轨迹ID 和 时间t
        # 这种简单的遍历在大数据集下效率低，可以用 bisect 优化，这里为了清晰用循环
        traj_idx = 0
        current_t = idx
        for t in self.trajectories:
            if current_t < len(t):
                break
            current_t -= len(t)
            traj_idx += 1
        
        # 获取整条轨迹
        full_traj = self.trajectories[traj_idx] # Shape: [Total_Len, 9]
        total_len = len(full_traj)
        
        # 2. 轨迹切片 (Slicing) & 填充 (Padding)
        # 我们需要从 current_t 开始，取 horizon 长度的数据
        end_t = current_t + self.horizon
        
        if end_t <= total_len:
            # 正常切片
            traj_seq = full_traj[current_t : end_t].copy()
        else:
            # 超出轨迹长度，用最后一个点填充 (Padding with terminal state)
            # 这教会模型：到了终点就停在那里
            real_part = full_traj[current_t:]
            pad_len = self.horizon - len(real_part)
            last_state = full_traj[-1]
            padding = np.tile(last_state, (pad_len, 1))
            traj_seq = np.vstack([real_part, padding])
        
        # 3. 坐标转换: Target-Centric (关键!)
        # 模型的任务是：给定当前相对位置，预测如何归零
        # 真实的着陆点 (Global Goal)
        global_goal_pos = full_traj[-1, :3]
        
        # 将轨迹序列转换为相对于着陆点的坐标
        # 注意：只转换位置(前3维)，速度和姿态是矢量的，不需要减去位置
        traj_seq[:, :3] -= global_goal_pos
        
        # 4. 归一化
        traj_seq_norm = self.normalize(traj_seq)
        
        # 5. 加载对应的图像
        # 假设图像命名格式: trajectory_00/frame_000.png
        # 你的 JCurveGenerator 生成的ID是整数
        traj_name = f"trajectory_{traj_idx:02d}" 
        frame_name = f"frame_{current_t:03d}.png" # 假设你有这个
        img_path = os.path.join(self.image_root, traj_name, frame_name)
        
        # 如果没有图像文件（测试代码时），生成全黑图以防报错
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        else:
            # print(f"Warning: Image not found {img_path}, using black image.")
            image = torch.zeros(3, 256, 256)

        # 6. 构建 Condition (模型的输入条件)
        # Condition = 当前的相对状态 (Normalized)
        condition_state = traj_seq_norm[0] 
        
        # 返回 PyTorch Tensor
        return {
            'trajectory': torch.FloatTensor(traj_seq_norm), # [Horizon, 9] - Ground Truth
            'condition': torch.FloatTensor(condition_state),# [9] - Start State
            'image': image,                                 # [3, H, W] - Visual Context
            'global_goal': torch.FloatTensor(global_goal_pos) # [3] - 用于调试或重构绝对坐标
        }

# ================= 使用测试 =================
if __name__ == "__main__":
    # 假设你的数据存放在这里
    data_path = 'data/j_curve_trajectories'
    img_path = 'data/images' # 哪怕它是空的也没关系，代码会生成黑图
    
    # 实例化数据集
    dataset = LunarLandingDataset(data_path, img_path, horizon=128, include_velocity=True)
    
    print(f"Total samples: {len(dataset)}")
    
    # 取一个样本看看
    sample = dataset[0]
    traj = sample['trajectory']
    cond = sample['condition']
    
    print("Trajectory Shape:", traj.shape) # 应该是 [128, 9]
    print("Condition Shape:", cond.shape)  # 应该是 [9]
    print("Start Relative Pos (Norm):", cond[:3])
    print("End Relative Pos (Norm):", traj[-1, :3]) # 应该非常接近归一化后的0
    
    # 验证是否收敛到0 (即 Target-Centric 是否生效)
    # 反归一化最后一个点的位置
    last_pt_norm = traj[-1].numpy()
    last_pt_phys = last_pt_norm * dataset.stats['std'] + dataset.stats['mean']
    print("Physical End Position (Should be near 0,0,0):", last_pt_phys[:3])