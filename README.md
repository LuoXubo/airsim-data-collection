# AirSim Data Collection

This project is based on AirSim and Unreal Engine 5 (UE5), providing a set of tools for generating lunar landing trajectories and automatically collecting images with pose ground truth data.

## ✨ Features

*   **Trajectory Generation**:
    *   `generators/generate_landing_path.py`: Generates three-stage landing trajectories using spline interpolation with sinusoidal noise for diversity.
    *   `generators/generate_landing_path_J_200f.py`: Generates J-curve trajectories (Cruise → Turn → Vertical Descent) simulating lunar landing profiles.

*   **Automated Data Collection (`capture_from_path_pkl.py`)**:
    *   Automatically connects to the AirSim client.
    *   Reads generated trajectory files and controls the drone to fly along the predefined path.
    *   Synchronously collects RGB images and pose data.
    *   Saves data in a standardized format suitable for deep learning training.

## 🛠️ Requirements

*   Python 3.x
*   AirSim (Requires UE5 environment)
*   Python Libraries:
    ```bash
    pip install numpy matplotlib opencv-python msgpack-rpc-python
    ```

## 📁 Project Structure

```text
airsim-data-collection/
├── airsim/                         # Local AirSim client library
├── generators/                     # Trajectory generation scripts
│   ├── generate_landing_path.py    # Spline-based trajectory generator
│   └── generate_landing_path_J_200f.py  # J-curve trajectory generator
├── vis/                            # Visualization tools
│   └── vis.html                    # 3D trajectory visualization
├── data/                           # Data directory
│   ├── trajectories/               # Generated trajectory files (.txt)
│   └── captured_images_trajectories/  # Collected image + pose data
├── capture_from_path_pkl.py        # Main data collection script
├── collect_from_path.py            # Legacy collection script (txt format)
└── setup_path.py                   # AirSim module path configuration
```

## 🚀 Usage

### 1. Generate Trajectories
Run the trajectory generation script:
```bash
python generators/generate_landing_path_J_200f.py
```
This will create trajectory files (`.txt`) in the `data/trajectories` directory.

### 2. Collect Data
Ensure the AirSim simulation environment (UE5) is running, then run:
```bash
python capture_from_path_pkl.py
```
The script will read trajectory files from `data/trajectories`, control the drone flight, and save data to `data/captured_images_trajectories`.

## 📂 Output Data Structure

Data is saved in the following format, designed for deep learning training:

```text
data/captured_images_trajectories/
├── trajectory_1/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── 2.jpg
│   ├── ...
│   └── traj_data.pkl
├── trajectory_2/
│   ├── 0.jpg
│   ├── 1.jpg
│   ├── ...
│   └── traj_data.pkl
└── ...
```

### traj_data.pkl Format

```python
import pickle
import numpy as np

# Load trajectory data
with open("trajectory_1/traj_data.pkl", "rb") as f:
    traj_data = pickle.load(f)

# Data structure:
# traj_data = {
#     "position": np.array([[x, y, z], ...]),      # shape: (T, 3)
#     "orientation": np.array([[w, x, y, z], ...]) # shape: (T, 4), quaternion
# }
```

| Field | Shape | Description |
|-------|-------|-------------|
| `position` | (T, 3) | XYZ coordinates in meters |
| `orientation` | (T, 4) | Quaternion [w, x, y, z] |

## 📝 Script Details

*   **`generators/generate_landing_path.py`**: Uses Hermite spline interpolation to generate smooth three-stage landing paths with customizable stage ratios and pitch angles.
*   **`generators/generate_landing_path_J_200f.py`**: Implements J-curve trajectory generation with horizontal cruise, smooth turn, and vertical descent phases.
*   **`capture_from_path_pkl.py`**: Uses `airsim.VehicleClient` to control the drone, captures images via `simGetImages`, and saves pose data as pickle files.

## ⚠️ Notes
*   Ensure AirSim's `settings.json` is correctly configured to allow API control.
*   During collection, the script forces drone position (`ignore_collision=True`). Ensure there are no impassable obstacles in the scene.
*   Images are saved as JPEG for storage efficiency. Modify the script if PNG is required.
