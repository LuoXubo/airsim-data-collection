import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

def generate_trajectory_with_orientation(start, end, num_points=1000, noise=5.0, shape_seed=0, 
                                          pitch_stage1_deg=-60, pitch_stage2_range=(-60, -90), 
                                          pitch_stage3_deg=-90,
                                          stage_ratios=(0.4, 0.3, 0.3)):
    """
    Generate three-stage trajectory with orientation.
    
    Parameters:
    - start, end: start and end positions [x, y, z]
    - num_points: total number of points
    - noise: amplitude of sinusoidal deviation
    - shape_seed: random seed for shape variation
    - pitch_stage1_deg: pitch angle for stage 1 (cruise)
    - pitch_stage2_range: (start, end) pitch angles for stage 2 (transition)
    - pitch_stage3_deg: pitch angle for stage 3 (landing)
    - stage_ratios: (ratio1, ratio2, ratio3) proportion of points in each stage
    """
    np.random.seed(shape_seed)
    start = np.array(start, dtype=float)
    end = np.array(end, dtype=float)
    
    # Calculate number of points per stage
    num_stage1 = int(num_points * stage_ratios[0])
    num_stage2 = int(num_points * stage_ratios[1])
    num_stage3 = num_points - num_stage1 - num_stage2
    
    # Calculate stage transition positions
    stage1_end_pos = start + (end - start) * 0.5  # 50% of the way
    stage2_end_pos = start + (end - start) * 0.8  # 80% of the way
    stage3_end_pos = end
    
    # Initialize arrays
    points = []
    pitches = []
    
    # === Stage 1: Cruise (horizontal movement, constant pitch) ===
    t1 = np.linspace(0, 1, num_stage1)
    stage1_points = np.outer(1 - t1, start) + np.outer(t1, stage1_end_pos)
    # Keep z constant in stage 1
    stage1_points[:, 2] = start[2]
    
    # Add sinusoidal variation
    direction1 = stage1_end_pos - start
    direction1 = direction1 / (np.linalg.norm(direction1) + 1e-8)
    ortho1 = get_orthogonal_vectors(direction1)
    deviation1 = add_sinusoidal_deviation(t1, ortho1, noise, shape_seed)
    stage1_points += deviation1
    
    # Constant pitch for stage 1
    stage1_pitch = np.full(num_stage1, np.deg2rad(pitch_stage1_deg))
    
    points.append(stage1_points)
    pitches.append(stage1_pitch)
    
    # === Stage 2: Transition (descending + pitch change) ===
    t2 = np.linspace(0, 1, num_stage2)
    stage2_points = np.outer(1 - t2, stage1_end_pos) + np.outer(t2, stage2_end_pos)
    # Quadratic descent in z
    z_start = stage1_end_pos[2]
    z_end = stage2_end_pos[2]
    stage2_points[:, 2] = z_start + (z_end - z_start) * (t2 ** 2)
    
    # Add sinusoidal variation
    direction2 = stage2_end_pos - stage1_end_pos
    direction2 = direction2 / (np.linalg.norm(direction2) + 1e-8)
    ortho2 = get_orthogonal_vectors(direction2)
    deviation2 = add_sinusoidal_deviation(t2, ortho2, noise * 0.5, shape_seed + 1)
    stage2_points += deviation2
    
    # Linear pitch change from pitch_stage2_range[0] to pitch_stage2_range[1]
    stage2_pitch = np.linspace(np.deg2rad(pitch_stage2_range[0]), 
                               np.deg2rad(pitch_stage2_range[1]), 
                               num_stage2)
    
    points.append(stage2_points)
    pitches.append(stage2_pitch)
    
    # === Stage 3: Landing (minimal movement, constant pitch) ===
    t3 = np.linspace(0, 1, num_stage3)
    stage3_points = np.outer(1 - t3, stage2_end_pos) + np.outer(t3, stage3_end_pos)
    
    # Slow vertical descent
    stage3_points[:, 2] = stage2_end_pos[2] + (stage3_end_pos[2] - stage2_end_pos[2]) * t3
    
    # Minimal horizontal movement (can add small oscillation)
    stage3_points[:, 0] = stage2_end_pos[0]  # Keep x constant
    stage3_points[:, 1] = stage2_end_pos[1]  # Keep y constant
    
    # Constant pitch for stage 3
    stage3_pitch = np.full(num_stage3, np.deg2rad(pitch_stage3_deg))
    
    points.append(stage3_points)
    pitches.append(stage3_pitch)
    
    # Combine all stages
    all_points = np.vstack(points)
    all_pitches = np.concatenate(pitches)
    
    # Roll and yaw are zero
    roll = np.zeros_like(all_pitches)
    yaw = np.zeros_like(all_pitches)
    
    # Combine position and orientation: [x, y, z, roll, pitch, yaw]
    trajectory = np.hstack([all_points, roll[:, None], all_pitches[:, None], yaw[:, None]])
    return trajectory

def get_orthogonal_vectors(direction):
    """Get two orthogonal vectors to the given direction."""
    if not np.allclose(direction, [0, 0, 1]):
        ortho1 = np.cross(direction, [0, 0, 1])
    else:
        ortho1 = np.cross(direction, [0, 1, 0])
    ortho1 = ortho1 / (np.linalg.norm(ortho1) + 1e-8)
    ortho2 = np.cross(direction, ortho1)
    ortho2 = ortho2 / (np.linalg.norm(ortho2) + 1e-8)
    return ortho1, ortho2

def add_sinusoidal_deviation(t, ortho_vectors, noise, seed):
    """Add sinusoidal deviation in the plane defined by orthogonal vectors."""
    np.random.seed(seed)
    ortho1, ortho2 = ortho_vectors
    
    freq1 = np.random.uniform(1, 3)
    freq2 = np.random.uniform(1, 3)
    amp1 = np.random.uniform(0.3, 0.7) * noise
    amp2 = np.random.uniform(0.3, 0.7) * noise
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    
    deviation = (amp1 * np.sin(2 * np.pi * freq1 * t + phase1))[:, None] * ortho1 + \
                (amp2 * np.cos(2 * np.pi * freq2 * t + phase2))[:, None] * ortho2
    return deviation

def save_trajectory(points, filename):
    np.savetxt(filename, points, fmt="%.3f", delimiter=",")

def visualize_trajectories(trajectories, show_orientation=True):
    """Visualize trajectories with optional orientation indicators."""
    fig = plt.figure(figsize=(12, 5))
    
    # 3D trajectory plot
    ax1 = fig.add_subplot(121, projection='3d')
    for i, traj in enumerate(trajectories):
        ax1.plot(traj[:, 0], traj[:, 1], traj[:, 2], label=f'Trajectory {i+1}')
        # Mark stages with different colors
        num_points = len(traj)
        stage1_end = int(num_points * 0.4)
        stage2_end = int(num_points * 0.7)
        ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='Start')
        ax1.scatter(traj[stage1_end, 0], traj[stage1_end, 1], traj[stage1_end, 2], 
                   c='yellow', s=100, marker='s', label='Stage 1→2')
        ax1.scatter(traj[stage2_end, 0], traj[stage2_end, 1], traj[stage2_end, 2], 
                   c='orange', s=100, marker='^', label='Stage 2→3')
        ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='*', label='End')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.legend()
    ax1.set_title('3D Trajectory')
    
    # Pitch angle plot
    ax2 = fig.add_subplot(122)
    for i, traj in enumerate(trajectories):
        pitch_deg = np.rad2deg(traj[:, 4])
        ax2.plot(pitch_deg, label=f'Trajectory {i+1}')
    ax2.set_xlabel('Point Index')
    ax2.set_ylabel('Pitch Angle (degrees)')
    ax2.set_title('Pitch Angle vs Time')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    save_path = 'data/trajs'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Define start and end positions
    start = (0, 0, 0)
    end = (400, 0, 50)
    
    # Generate trajectories
    num_trajectories = 1
    trajectories = []
    
    for i in range(num_trajectories):
        traj = generate_trajectory_with_orientation(
            start, end, 
            num_points=600,  # Total points
            noise=5.0,  # Deviation amplitude
            shape_seed=i,  # Different seed for each trajectory
            pitch_stage1_deg=-60,  # Cruise pitch
            pitch_stage2_range=(-60, -90),  # Transition pitch range
            pitch_stage3_deg=-90,  # Landing pitch (looking down)
            stage_ratios=(0.4, 0.3, 0.3)  # 40% cruise, 30% transition, 30% landing
        )
        save_name = f'trajectory_{i:02d}.txt'
        save_trajectory(traj, f"{save_path}/{save_name}")
        trajectories.append(traj)
        print(f"Generated trajectory {i+1}: {len(traj)} points")
    
    # Visualize all trajectories
    visualize_trajectories(trajectories)