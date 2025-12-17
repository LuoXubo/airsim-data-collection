import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
from tqdm import tqdm  # 如果没有请 pip install tqdm

# ==========================================
# 核心数学引擎：五次多项式规划 (Minimum Jerk)
# ==========================================
class QuinticPolynomial:
    """
    计算满足边界约束的平滑曲线:
    p(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
    
    保证起点和终点的 位置(x), 速度(v), 加速度(a) 严格满足约束
    """
    def __init__(self, x0, v0, a0, x1, v1, a1, T):
        self.a0 = x0
        self.a1 = v0
        self.a2 = a0 / 2.0
        
        A = np.array([[T**3, T**4, T**5],
                      [3*T**2, 4*T**3, 5*T**4],
                      [6*T, 12*T**2, 20*T**3]])
        b = np.array([x1 - self.a0 - self.a1*T - self.a2*T**2,
                      v1 - self.a1 - 2*self.a2*T,
                      a1 - 2*self.a2])
        
        try:
            x = np.linalg.solve(A, b)
            self.a3, self.a4, self.a5 = x[0], x[1], x[2]
        except np.linalg.LinAlgError:
            print("Error: Matrix singularity in calculation.")
            self.a3 = self.a4 = self.a5 = 0

    def calc_point(self, t):
        return self.a0 + self.a1*t + self.a2*t**2 + self.a3*t**3 + self.a4*t**4 + self.a5*t**5

    def calc_vel(self, t):
        return self.a1 + 2*self.a2*t + 3*self.a3*t**2 + 4*self.a4*t**3 + 5*self.a5*t**4

    def calc_acc(self, t):
        return 2*self.a2 + 6*self.a3*t + 12*self.a4*t**2 + 20*self.a5*t**3

# ==========================================
# 轨迹生成器 (生成单条专家轨迹)
# ==========================================
def generate_expert_trajectory(start_pos, start_vel, start_acc, target_pos, duration, dt=0.1):
    """
    生成包含位置、速度、姿态的完整轨迹
    """
    # 规划三个轴
    # 终点约束：位置=target, 速度=0, 加速度=0 (软着陆)
    qp_x = QuinticPolynomial(start_pos[0], start_vel[0], start_acc[0], target_pos[0], 0, 0, duration)
    qp_y = QuinticPolynomial(start_pos[1], start_vel[1], start_acc[1], target_pos[1], 0, 0, duration)
    qp_z = QuinticPolynomial(start_pos[2], start_vel[2], start_acc[2], target_pos[2], 0, 0, duration)
    
    times = np.arange(0, duration + dt, dt)
    traj_data = [] # [x, y, z, vx, vy, vz, roll, pitch, yaw]
    
    for t in times:
        # 1. 计算平动状态
        px = qp_x.calc_point(t)
        py = qp_y.calc_point(t)
        pz = qp_z.calc_point(t)
        
        vx = qp_x.calc_vel(t)
        vy = qp_y.calc_vel(t)
        vz = qp_z.calc_vel(t)
        
        ax = qp_x.calc_acc(t)
        ay = qp_y.calc_acc(t)
        az = qp_z.calc_acc(t)
        
        # 2. 计算姿态 (Simplified Guidance Logic)
        # 假设：
        # Yaw (偏航角) -> 始终朝向目标方向或沿着速度矢量
        # Pitch (俯仰角) -> 由推力方向决定。
        #   - 高速水平飞行时，Pitch 较小 (接近0度，推力水平分量大)
        #   - 垂直下降时，Pitch = -90度 (推力全垂直)
        
        # 计算水平速度
        v_horiz = np.sqrt(vx**2 + vy**2)
        
        # Yaw: 简单的对准速度方向
        if v_horiz > 0.1:
            yaw = np.arctan2(vy, vx)
        else:
            yaw = 0.0 # 几乎静止时保持
            
        # Pitch: 启发式算法，模拟着陆器的“抬头”过程
        # 定义：0度为水平，-90度为垂直向下
        # 逻辑：基于剩余距离和当前高度的比例，或者基于速度矢量
        
        # 方案：使用反正切模拟从水平到垂直的过渡
        # 当 v_horiz 很大时，pitch -> 0
        # 当 v_horiz 很小时，pitch -> -pi/2
        pitch = np.arctan2(vz, v_horiz + 0.001) # 沿着速度矢量
        
        # 修正：着陆器不仅要沿着速度飞，还要反推，所以通常推力方向与速度方向相反
        # 但对于轨迹数据集，我们记录机体姿态。假设机体轴线与推力方向一致。
        # 限制 pitch 范围，防止翻跟头
        pitch = np.clip(pitch, -np.pi/2, 0)
        
        roll = 0.0 # 简化，假设无滚转
        
        traj_data.append([px, py, pz, vx, vy, vz, roll, pitch, yaw])
        
    return np.array(traj_data)

# ==========================================
# 数据集构建器 (包含场景随机化)
# ==========================================
def create_dataset(output_dir, num_samples=1000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Start generating {num_samples} trajectories in '{output_dir}'...")
    
    # 场景比例分布
    scenarios = ['approach'] * int(num_samples * 0.4) + \
                ['dogleg'] * int(num_samples * 0.4) + \
                ['terminal'] * int(num_samples * 0.2)
    np.random.shuffle(scenarios)
    
    pbar = tqdm(total=num_samples)
    
    for i, scenario in enumerate(scenarios):
        # --- 核心：Target-Centric 坐标系 ---
        # 无论我们在哪里，目标永远是 (0,0,0)
        target_pos = [0, 0, 0]
        
        # 初始化变量
        start_pos = [0,0,0]
        start_vel = [0,0,0]
        start_acc = [0,0,0] # 初始加速度通常设为0，或模拟重力
        duration = 0
        
        # --- 场景 1: 高空进近 (Approach) ---
        # 特点：高度高，距离远，有较大的初始水平速度
        if scenario == 'approach':
            start_z = np.random.uniform(1500, 3000) # 1.5km - 3km 高度
            dist_horiz = np.random.uniform(500, 2000) # 水平距离
            angle = np.random.uniform(0, 2*np.pi)
            
            start_pos = [
                dist_horiz * np.cos(angle),
                dist_horiz * np.sin(angle),
                start_z
            ]
            
            # 初始速度：朝着目标方向，带有一定的下沉速度
            speed_h = np.random.uniform(30, 60) # 30-60 m/s
            start_vel = [
                -np.cos(angle) * speed_h, # 负号表示飞向原点
                -np.sin(angle) * speed_h,
                np.random.uniform(-10, -30) # 垂直速度
            ]
            
            duration = np.random.uniform(40, 70) # 飞行时间较长
            
        # --- 场景 2: 侧向规避/重规划 (Dog-leg) ---
        # 特点：这是对 Diffusion 最重要的训练数据！
        # 初始位置可能离目标很近（垂直投影），但有偏差，必须“拐弯”去原点
        elif scenario == 'dogleg':
            start_z = np.random.uniform(300, 1000)
            # 侧向偏差：明明在 500米高空，却偏离了目标 200米
            offset_x = np.random.uniform(-300, 300)
            offset_y = np.random.uniform(-300, 300)
            start_pos = [offset_x, offset_y, start_z]
            
            # 初始速度：可能是垂直向下的（之前想去正下方），突然要改道去 (0,0,0)
            start_vel = [
                np.random.normal(0, 2), # 水平速度很小
                np.random.normal(0, 2),
                np.random.uniform(-10, -20)
            ]
            
            duration = np.random.uniform(20, 40)
            
        # --- 场景 3: 垂直着陆段 (Terminal) ---
        # 特点：低空，就在目标正上方附近，主要任务是垂直减速
        elif scenario == 'terminal':
            start_z = np.random.uniform(50, 300)
            offset_r = np.random.uniform(0, 20) # 偏差很小
            angle = np.random.uniform(0, 2*np.pi)
            
            start_pos = [
                offset_r * np.cos(angle),
                offset_r * np.sin(angle),
                start_z
            ]
            
            start_vel = [
                np.random.uniform(-2, 2),
                np.random.uniform(-2, 2),
                np.random.uniform(-2, -10)
            ]
            duration = np.random.uniform(10, 25)
            
        # 生成轨迹
        traj = generate_expert_trajectory(start_pos, start_vel, start_acc, target_pos, duration)
        
        # 数据格式化保存
        # Filename: traj_{ID}_{Scenario}.txt
        filename = f"traj_{i:04d}_{scenario}.txt"
        filepath = os.path.join(output_dir, filename)
        
        # 保存为 CSV 格式: x, y, z, vx, vy, vz, roll, pitch, yaw
        np.savetxt(filepath, traj, fmt="%.6f", delimiter=",")
        
        pbar.update(1)
        
    pbar.close()
    print("Dataset generation complete.")

# ==========================================
# 可视化工具 (验证数据质量)
# ==========================================
def visualize_generated_data(data_dir, num_show=10):
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    if not files:
        print("No data found to visualize.")
        return
        
    np.random.shuffle(files)
    selected = files[:num_show]
    
    fig = plt.figure(figsize=(14, 6))
    
    # Plot 1: 3D Trajectories
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(0, 0, 0, c='r', marker='*', s=200, label='Target (0,0,0)')
    
    # Plot 2: Altitude vs Pitch (check dynamics)
    ax2 = fig.add_subplot(122)
    
    for f in selected:
        path = os.path.join(data_dir, f)
        data = np.loadtxt(path, delimiter=',')
        
        # Data columns: 0:x, 1:y, 2:z, ... 7:pitch
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        pitch = np.rad2deg(data[:, 7])
        
        # Get scenario name from filename for label
        label = f.split('_')[2].split('.')[0]
        
        # 3D Plot
        ax1.plot(x, y, z, label=label, alpha=0.7)
        ax1.scatter(x[0], y[0], z[0], s=10) # start point
        
        # Pitch Plot
        ax2.plot(z, pitch, alpha=0.6)
        
    ax1.set_title('3D Trajectories (Target-Centric)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Altitude (m)')
    # ax1.legend()
    
    ax2.set_title('Pitch Angle vs Altitude')
    ax2.set_xlabel('Altitude (m)')
    ax2.set_ylabel('Pitch (deg) [0=Horiz, -90=Vert]')
    ax2.invert_xaxis() # High altitude on left, landing on right
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 1. 定义保存路径
    DATA_PATH = "data\\lunar_landing_dataset_v1"
    
    # 2. 生成数据
    # 建议生成至少 1000-5000 条用于初步训练
    create_dataset(DATA_PATH, num_samples=50) 
    
    # 3. 可视化检查
    visualize_generated_data(DATA_PATH, num_show=15)