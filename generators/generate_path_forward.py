import os
import numpy as np


class ForwardTrajectoryGenerator:
	"""Generate forward straight-line trajectories with yaw=0 and pitch=-90."""

	def __init__(self, save_dir='trajectories'):
		self.save_dir = save_dir
		if not os.path.exists(save_dir):
			os.makedirs(save_dir)

	def generate_trajectory(
		self,
		start_x=0.0,
		start_y=0.0,
		start_z=0.0,
		distance=300.0,
		altitude=10.0,
		num_points=200,
	):
		"""
		Build a straight path along +X with constant altitude.

		Altitude follows the project convention: larger z means lower (downward positive).
		Pitch is fixed to -90 degrees (nose down), yaw fixed to 0.
		"""

		end_x = start_x + distance
		x_vals = np.linspace(start_x, end_x, num_points)
		y_vals = np.full(num_points, start_y)
		z_vals = np.full(num_points, altitude if altitude is not None else start_z)

		# roll, pitch, yaw (degrees); pitch=-90 (nose down), yaw=0
		roll = np.zeros(num_points)
		pitch = np.full(num_points, -90.0)
		yaw = np.zeros(num_points)

		# Store orientations as [roll, pitch, yaw] for saving
		positions = np.column_stack([x_vals, y_vals, z_vals])
		orientations = np.column_stack([roll, pitch, yaw])

		return {
			'positions': positions,
			'orientations': orientations,
		}

	def generate_multiple_trajectories(self, num_trajectories=10, **kwargs):
		"""Generate multiple straight trajectories with small random offsets."""

		trajectories = []
		for i in range(num_trajectories):
			params = {
				'start_x': np.random.uniform(-5, 5),
				'start_y': np.random.uniform(-10, 10),
				'start_z': 0.0,
				'distance': np.random.uniform(250, 350),
				'altitude': np.random.uniform(8, 12),
				'num_points': np.random.randint(90, 110),
			}

			params.update(kwargs)
			traj = self.generate_trajectory(**params)
			traj['id'] = i
			traj['params'] = params
			trajectories.append(traj)
			print(
				f"Trajectory {i+1}/{num_trajectories}: start=({params['start_x']:.1f}, {params['start_y']:.1f}), "
				f"distance={params['distance']:.1f}, altitude={params['altitude']:.1f}"
			)

		return trajectories

	def save_trajectories(self, trajectories):
		"""Save trajectories as txt with columns x,y,z,roll,pitch,yaw (radians)."""

		for traj in trajectories:
			traj_id = traj.get('id', 0)
			positions = np.asarray(traj['positions'])
			orientations = np.asarray(traj['orientations'])

			roll_rad = np.deg2rad(orientations[:, 0])
			pitch_rad = np.deg2rad(orientations[:, 1])
			yaw_rad = np.deg2rad(orientations[:, 2])

			data = np.column_stack([
				positions[:, 0],
				positions[:, 1],
				positions[:, 2],
				roll_rad,
				pitch_rad,
				yaw_rad,
			])

			filepath = os.path.join(self.save_dir, f"trajectory_{traj_id:02d}.txt")
			np.savetxt(filepath, data, fmt='%.3f', delimiter=',')
			print(f"Saved: {filepath}")


if __name__ == '__main__':
	generator = ForwardTrajectoryGenerator(save_dir='data\\trajectories_forward')

	num_trajectories = 1
	trajectories = generator.generate_multiple_trajectories(num_trajectories=num_trajectories)
	generator.save_trajectories(trajectories)

	print(f"Complete. Generated {num_trajectories} forward trajectories with yaw=0°, pitch=-90°.")
