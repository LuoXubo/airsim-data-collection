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
    Completely smooth trajectory using global spline-like interpolation.
    
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
    
    idx1 = num_stage1
    idx2 = num_stage1 + num_stage2
    
    # Global time parameter
    t_global = np.linspace(0, 1, num_points)
    
    # --- Generate Base Trajectory using Global Smooth Functions ---
    
    # X: Simple linear from start to end (no discontinuity possible)
    x_vals = start[0] + (end[0] - start[0]) * t_global
    
    # Y: Linear (or constant if start[1] == end[1])
    y_vals = start[1] + (end[1] - start[1]) * t_global
    
    # Z: Piecewise smooth with C1 continuity
    # Stage 1: constant Z = start[2]
    # Stage 2: smooth transition from start[2] to end[2] * 0.3 (some intermediate height)
    # Stage 3: linear descent to end[2]
    
    # Key points for Z:
    # t=0: z=start[2], slope=0
    # t=t1 (end of stage 1): z=start[2], slope=0
    # t=t2 (end of stage 2): z=z_intermediate, slope=final_slope
    # t=1: z=end[2]
    
    t1 = stage_ratios[0]  # normalized time for end of stage 1
    t2 = stage_ratios[0] + stage_ratios[1]  # normalized time for end of stage 2
    
    z_start = start[2]
    z_end = end[2]
    # Intermediate Z at end of stage 2 (proportional)
    z_intermediate = z_start + (z_end - z_start) * 0.3
    
    z_vals = np.zeros(num_points)
    
    for i, t in enumerate(t_global):
        if t <= t1:
            # Stage 1: constant
            z_vals[i] = z_start
        elif t <= t2:
            # Stage 2: Hermite spline from (t1, z_start, slope=0) to (t2, z_intermediate, slope=final_slope)
            # Normalize to [0, 1] within this stage
            s = (t - t1) / (t2 - t1)
            
            # Target slope at end: should match Stage 3 linear slope
            # Stage 3 goes from z_intermediate to z_end over time (1 - t2)
            # slope = (z_end - z_intermediate) / (1 - t2)
            final_slope = (z_end - z_intermediate) / (1 - t2) if (1 - t2) > 0 else 0
            
            # Convert slope to local parameter space
            local_slope = final_slope * (t2 - t1)
            
            # Hermite basis functions
            h00 = 2*s**3 - 3*s**2 + 1
            h10 = s**3 - 2*s**2 + s
            h01 = -2*s**3 + 3*s**2
            h11 = s**3 - s**2
            
            # p0 = z_start, m0 = 0 (flat entry)
            # p1 = z_intermediate, m1 = local_slope
            z_vals[i] = h00 * z_start + h10 * 0 + h01 * z_intermediate + h11 * local_slope
        else:
            # Stage 3: linear descent
            s = (t - t2) / (1 - t2) if (1 - t2) > 0 else 1
            z_vals[i] = z_intermediate + (z_end - z_intermediate) * s
    
    # --- Generate Pitch using Global Smooth Function ---
    pitch_start = np.deg2rad(pitch_stage1_deg)
    pitch_end = np.deg2rad(pitch_stage3_deg)
    
    pitch_vals = np.zeros(num_points)
    
    for i, t in enumerate(t_global):
        if t <= t1:
            # Stage 1: constant pitch
            pitch_vals[i] = pitch_start
        elif t <= t2:
            # Stage 2: Smooth transition using smoothstep (C1 continuous)
            s = (t - t1) / (t2 - t1)
            # Smootherstep: 6s^5 - 15s^4 + 10s^3 (C2 continuous)
            smooth_s = 6*s**5 - 15*s**4 + 10*s**3
            pitch_vals[i] = pitch_start + (pitch_end - pitch_start) * smooth_s
        else:
            # Stage 3: constant pitch
            pitch_vals[i] = pitch_end
    
    # Stack base trajectory
    all_points = np.column_stack((x_vals, y_vals, z_vals))
    all_pitches = pitch_vals.copy()
    
    # --- Apply Global Continuous Noise (Position) ---
    # Calculate orthogonal vectors based on main direction
    main_dir = end - start
    ortho1, ortho2 = get_orthogonal_vectors(main_dir)
    
    # Generate random noise parameters with LOWER frequency for smoother curves
    freq1 = np.random.uniform(0.5, 1.5)
    freq2 = np.random.uniform(0.5, 1.5)
    phase1 = np.random.uniform(0, 2 * np.pi)
    phase2 = np.random.uniform(0, 2 * np.pi)
    
    # Generate continuous noise signal
    raw_noise = (np.sin(2 * np.pi * freq1 * t_global + phase1))[:, None] * ortho1 + \
                (np.cos(2 * np.pi * freq2 * t_global + phase2))[:, None] * ortho2
                
    # Create Noise Amplitude Envelope using SMOOTH function
    # Use smootherstep for ALL transitions
    amp = np.zeros(num_points)
    
    for i, t in enumerate(t_global):
        if t <= t1 * 0.2:
            # Fade in at very start
            s = t / (t1 * 0.2)
            amp[i] = noise * (6*s**5 - 15*s**4 + 10*s**3)
        elif t <= t1:
            # Full noise in stage 1
            amp[i] = noise
        elif t <= t2:
            # Smooth transition to low noise
            s = (t - t1) / (t2 - t1)
            smooth_s = 6*s**5 - 15*s**4 + 10*s**3
            amp[i] = noise * (1 - 0.8 * smooth_s)  # From noise to 0.2*noise
        else:
            # Low noise in stage 3
            amp[i] = noise * 0.2
    
    # Apply noise to position
    all_points += raw_noise * amp[:, None]
    
    # --- Apply Global Continuous Noise (Pitch) ---
    # Lower frequency for smoother pitch variation
    pitch_noise_amp = np.deg2rad(2.0)
    pitch_noise = pitch_noise_amp * np.sin(2 * np.pi * 0.8 * t_global + phase1)
    
    # Pitch noise envelope: small in stage 1/2, larger in stage 3, all smooth
    pitch_amp = np.zeros(num_points)
    for i, t in enumerate(t_global):
        if t <= t2:
            pitch_amp[i] = 0.2
        else:
            # Smooth transition to full
            s = (t - t2) / (1 - t2) if (1 - t2) > 0 else 1
            smooth_s = 6*s**5 - 15*s**4 + 10*s**3
            pitch_amp[i] = 0.2 + 0.8 * smooth_s
    
    all_pitches += pitch_noise * pitch_amp
    
    # Final assembly: [x, y, z, roll, pitch, yaw]
    roll = np.zeros_like(all_pitches)
    yaw = np.zeros_like(all_pitches)
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
        # ax1.scatter(traj[0, 0], traj[0, 1], traj[0, 2], c='green', s=100, marker='o', label='Start')
        # ax1.scatter(traj[stage1_end, 0], traj[stage1_end, 1], traj[stage1_end, 2], 
        #            c='yellow', s=100, marker='s', label='Stage 1→2')
        # ax1.scatter(traj[stage2_end, 0], traj[stage2_end, 1], traj[stage2_end, 2], 
        #            c='orange', s=100, marker='^', label='Stage 2→3')
        # ax1.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], c='red', s=100, marker='*', label='End')
    
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
    num_trajectories = 15
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