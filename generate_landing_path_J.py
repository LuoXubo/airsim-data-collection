import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import json


class JCurveTrajectoryGenerator:
    """
    J型曲线轨迹生成器
    生成符合月面着陆特征的J型轨迹：水平段 -> 转弯段 -> 垂直段
    注意：仿真系统中z值越大表示高度越低（向下为正）
    """
    
    def __init__(self, save_dir='trajectories'):
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def generate_trajectory(self, 
                        start_x=0, start_y=0, start_z=0,  # 起始位置
                        landing_x=300, landing_y=0,      # 着陆位置
                        horizontal_distance=300,  # 水平飞行距离
                        initial_height=0,         # 初始高度（z=0）
                        final_height=10,          # 最终高度（z值增加，实际下降）
                        turn_start_ratio=0.75,    # 转弯开始位置比例
                        num_points=600,           # 总采样点数
                        pitch_start=-60,          # 初始俯仰角（度）
                        pitch_end=-90):           # 最终俯仰角（度）
        """
        生成单条J型曲线轨迹
        
        参数:
        - horizontal_distance: 水平段飞行的总距离
        - initial_height: 起始高度（z=0表示最高）
        - final_height: 着陆高度（z值更大，表示下降）
        - turn_start_ratio: 转弯段开始的位置比例（0-1）
        - num_points: 轨迹点数
        - pitch_start: 初始俯仰角
        - pitch_end: 最终俯仰角
        """
        
        # 计算各阶段点数
        turn_start_x = start_x + horizontal_distance * turn_start_ratio
        num_stage1 = int(num_points * turn_start_ratio)  # 水平段
        num_stage2 = int(num_points * 0.2)                # 转弯段
        num_stage3 = num_points - num_stage1 - num_stage2  # 垂直段
        
        trajectory = {
            'positions': [],
            'orientations': [],
            'stages': []
        }
        
        # ===== 第一阶段：水平巡航段 =====
        stage1_x = np.linspace(start_x, turn_start_x, num_stage1)
        stage1_z = np.ones(num_stage1) * start_z  # 保持初始高度
        stage1_y = np.zeros(num_stage1)  # Y方向保持为0
        stage1_pitch = np.ones(num_stage1) * pitch_start
        
        for i in range(num_stage1):
            trajectory['positions'].append([stage1_x[i], stage1_y[i], stage1_z[i]])
            trajectory['orientations'].append([stage1_pitch[i], 0, 0])  # pitch, roll, yaw
            trajectory['stages'].append(1)
        
        # ===== 第二阶段：转弯段（关键的J型弯曲） =====
        # 使用三次贝塞尔曲线来生成平滑的转弯
        t = np.linspace(0, 1, num_stage2)
        
        # X方向：从turn_start_x逐渐减慢到landing_x
        stage2_x = turn_start_x + (landing_x - turn_start_x) * (1 - (1-t)**2)
        
        # Z方向：使用三次函数从start_z加速下降到final_height
        stage2_z = start_z + final_height * (t**2)
        
        stage2_y = np.zeros(num_stage2)  # Y方向保持为0
        
        # 俯仰角从pitch_start平滑过渡到pitch_end
        stage2_pitch = pitch_start + (pitch_end - pitch_start) * t
        
        for i in range(num_stage2):
            trajectory['positions'].append([stage2_x[i], stage2_y[i], stage2_z[i]])
            trajectory['orientations'].append([stage2_pitch[i], 0, 0])
            trajectory['stages'].append(2)
        
        # ===== 第三阶段：垂直下降段 =====
        stage3_x = np.ones(num_stage3) * landing_x  # X保持在着陆点
        stage3_z = np.linspace(final_height, final_height + 10, num_stage3)  # 继续下降
        stage3_y = np.zeros(num_stage3)  # Y方向保持为0
        stage3_pitch = np.ones(num_stage3) * pitch_end  # 保持垂直
        
        for i in range(num_stage3):
            trajectory['positions'].append([stage3_x[i], stage3_y[i], stage3_z[i]])
            trajectory['orientations'].append([stage3_pitch[i], 0, 0])
            trajectory['stages'].append(3)
        
        return trajectory
    
    def generate_multiple_trajectories(self, num_trajectories=10, **kwargs):
        """
        生成多条轨迹，通过随机化起点、着陆点和其他参数增加多样性
        """
        trajectories = []
        
        for i in range(num_trajectories):
            # 随机化起点和着陆点
            rand_start_x = np.random.uniform(-10, 10)
            rand_start_y = 0  # Y方向保持为0
            rand_start_z = np.random.uniform(-5, 5)
            
            rand_landing_x = np.random.uniform(250, 400)
            rand_landing_y = 0  # Y方向保持为0
            
            # 随机化其他参数以增加轨迹多样性
            params = {
                'start_x': rand_start_x,
                'start_y': rand_start_y,
                'start_z': rand_start_z,
                'landing_x': rand_landing_x,
                'landing_y': rand_landing_y,
                'horizontal_distance': rand_landing_x - rand_start_x,  # 根据起点和着陆点计算
                'final_height': np.random.uniform(8, 12),
                'turn_start_ratio': np.random.uniform(0.7, 0.8),
                'num_points': np.random.randint(500, 700),
                'pitch_start': np.random.uniform(-65, -55),
                'pitch_end': -90
            }
            
            # 允许用户覆盖参数
            params.update(kwargs)
            
            trajectory = self.generate_trajectory(**params)
            trajectory['id'] = i
            trajectory['params'] = params
            trajectories.append(trajectory)
            
            print(f"Generated trajectory {i+1}/{num_trajectories}: Start({rand_start_x:.1f}, {rand_start_y:.1f}, {rand_start_z:.1f}) -> Landing({rand_landing_x:.1f}, {rand_landing_y:.1f})")
        
        return trajectories
    
    def save_trajectories(self, trajectories, filename='trajectories.json'):
        """保存轨迹到txt文件，格式与generate_landing_path.py一致"""
        for traj in trajectories:
            traj_id = traj['id']
            positions = np.array(traj['positions'])
            orientations = np.array(traj['orientations'])
            
            # 构建与generate_landing_path.py一致的格式: x, y, z, roll, pitch, yaw
            # orientations格式是 [pitch, roll, yaw]，需要转换为 [roll, pitch, yaw]
            # pitch需要转换为弧度
            roll = np.deg2rad(orientations[:, 1])  # roll
            pitch = np.deg2rad(orientations[:, 0])  # pitch
            yaw = np.deg2rad(orientations[:, 2])   # yaw
            
            trajectory_data = np.column_stack([
                positions[:, 0],  # x
                positions[:, 1],  # y
                positions[:, 2],  # z
                roll,             # roll
                pitch,            # pitch
                yaw               # yaw
            ])
            
            # 保存为txt文件
            filepath = os.path.join(self.save_dir, f'trajectory_{traj_id:02d}.txt')
            np.savetxt(filepath, trajectory_data, fmt="%.3f", delimiter=",")
            print(f"轨迹已保存到: {filepath}")
        
        print(f"共保存 {len(trajectories)} 条轨迹")
    
    def visualize_trajectory(self, trajectory, save_fig=False, fig_name='trajectory.png'):
        """可视化单条轨迹"""
        positions = np.array(trajectory['positions'])
        stages = np.array(trajectory['stages'])
        
        fig = plt.figure(figsize=(15, 5))
        
        # 3D视图
        ax1 = fig.add_subplot(131, projection='3d')
        colors = ['blue', 'orange', 'red']
        for stage in [1, 2, 3]:
            mask = stages == stage
            ax1.plot(positions[mask, 0], positions[mask, 1], positions[mask, 2], 
                    color=colors[stage-1], label=f'Stage {stage}', linewidth=2)
        ax1.set_xlabel('X (Horizontal Distance)')
        ax1.set_ylabel('Y (Horizontal Distance)')
        ax1.set_zlabel('Z (Altitude, Downward Positive)')
        ax1.legend()
        ax1.set_title('3D Trajectory View')
        ax1.invert_zaxis()  # 翻转Z轴使得向下为正更直观
        
        # 侧视图（X-Z平面，显示J型曲线）
        ax2 = fig.add_subplot(132)
        for stage in [1, 2, 3]:
            mask = stages == stage
            ax2.plot(positions[mask, 0], positions[mask, 2], 
                    color=colors[stage-1], label=f'Stage {stage}', linewidth=2)
        ax2.set_xlabel('X (Horizontal Distance)')
        ax2.set_ylabel('Z (Altitude, Downward Positive)')
        ax2.invert_yaxis()  # 翻转Y轴使得向下为正
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.set_title('Side View - J-Curve')
        
        # 俯仰角变化
        ax3 = fig.add_subplot(133)
        orientations = np.array(trajectory['orientations'])
        time_steps = np.arange(len(orientations))
        ax3.plot(time_steps, orientations[:, 0], linewidth=2)
        ax3.set_xlabel('Time Step')
        ax3.set_ylabel('Pitch Angle (degrees)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Pitch Angle Evolution')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.save_dir, fig_name), dpi=150)
            print(f"Figure saved to: {os.path.join(self.save_dir, fig_name)}")
        
        plt.show()
    
    def visualize_multiple_trajectories(self, trajectories, save_fig=False):
        """可视化多条轨迹"""
        fig = plt.figure(figsize=(15, 5))
        
        # 侧视图（X-Z平面）
        ax1 = fig.add_subplot(131)
        for traj in trajectories:
            positions = np.array(traj['positions'])
            ax1.plot(positions[:, 0], positions[:, 2], alpha=0.6, linewidth=1.5)
        ax1.set_xlabel('X (Horizontal Distance)')
        ax1.set_ylabel('Z (Altitude, Downward Positive)')
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.set_title(f'{len(trajectories)} J-Curve Trajectories (Side View)')
        
        # 3D视图
        ax2 = fig.add_subplot(132, projection='3d')
        for traj in trajectories:
            positions = np.array(traj['positions'])
            ax2.plot(positions[:, 0], positions[:, 1], positions[:, 2], alpha=0.6, linewidth=1)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.invert_zaxis()
        ax2.set_title('3D View')
        
        # 俯视图（X-Y平面）- 显示不同的起点和着陆点
        ax3 = fig.add_subplot(133)
        for traj in trajectories:
            positions = np.array(traj['positions'])
            # 标记起点和终点
            ax3.plot(positions[:, 0], positions[:, 1], alpha=0.5, linewidth=1.5)
            ax3.scatter(positions[0, 0], positions[0, 1], c='green', s=50, marker='o', alpha=0.7)  # 起点
            ax3.scatter(positions[-1, 0], positions[-1, 1], c='red', s=50, marker='x', alpha=0.7)  # 终点
        ax3.set_xlabel('X (Horizontal Distance)')
        ax3.set_ylabel('Y (Horizontal Distance)')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('Top View (Green: Start, Red: Landing)')
        ax3.legend(['Trajectories', 'Start Points', 'Landing Points'], loc='best')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(os.path.join(self.save_dir, 'multiple_trajectories.png'), dpi=150)
            print(f"Figure saved to: {os.path.join(self.save_dir, 'multiple_trajectories.png')}")
        
        plt.show()


# ===== 使用示例 =====
if __name__ == "__main__":
    # 创建生成器
    generator = JCurveTrajectoryGenerator(save_dir='data\\j_curve_trajectories')
    
    # 生成单条轨迹并可视化
    # print("Generating single trajectory example...")
    # single_trajectory = generator.generate_trajectory(
    #     start_x=0,
    #     start_y=0,
    #     start_z=0,
    #     landing_x=300,
    #     landing_y=0,
    #     final_height=10,
    #     num_points=600
    # )
    # generator.visualize_trajectory(single_trajectory, save_fig=True, fig_name='single_j_curve.png')
    
    # 生成多条轨迹
    print("\nGenerating multiple trajectories...")
    num_trajectories = 10  # 可以修改这个数字来控制生成轨迹的数量
    trajectories = generator.generate_multiple_trajectories(num_trajectories=num_trajectories)
    
    # 保存轨迹
    generator.save_trajectories(trajectories)
    
    # 可视化多条轨迹
    generator.visualize_multiple_trajectories(trajectories, save_fig=True)
    
    print(f"\nComplete! Generated {num_trajectories} J-curve trajectories with varying start and landing points")