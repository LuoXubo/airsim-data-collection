# AirSim Data Collection

This project is based on AirSim and Unreal Engine 5 (UE5), providing a set of tools for generating flight trajectories and automatically collecting images and pose ground truth data.

## ✨ Features

*   **Trajectory Generation (`generate_landing_path.py`)**:
    *   Generates three-stage landing trajectories (Cruise -> Transition -> Landing).
    *   Supports adding random sinusoidal noise to increase trajectory diversity.
    *   Automatically calculates orientation (Roll, Pitch, Yaw) along the path.
    *   Provides 3D trajectory visualization preview.

*   **Automated Collection (`collect_from_path.py`)**:
    *   Automatically connects to the AirSim client.
    *   Reads generated trajectory files and controls the drone to fly along the predefined path.
    *   Synchronously collects RGB images, pose ground truth, and timestamps.
    *   Saves data in separate folders for each trajectory with a standardized format.

## 🛠️ Requirements

*   Python 3.x
*   AirSim (Requires UE5 environment)
*   Python Libraries:
    ```bash
    pip install numpy matplotlib opencv-python msgpack-rpc-python
    ```

## 🚀 Usage

### 1. Generate Trajectories
Run the generation script to create trajectory files (`.txt`) in the `data/trajs` directory.
```bash
python generate_landing_path.py
```
*The script will display a 3D plot of the generated trajectory.*

### 2. Collect Data
Ensure the AirSim simulation environment (UE5) is running.
Run the collection script:
```bash
python collect_from_path.py
```
The script will sequentially read trajectory files from `data/trajs`, control the drone flight, and save data to `data/collected_from_path`.

## 📂 Output Data Structure

Data will be saved in the following structure:

```text
data/
├── trajs/                          # Generated trajectory files
│   ├── trajectory_0.txt
│   └── ...
└── collected_from_path/            # Collected data results
    └── collected_trajectory_0/     # Data for a specific trajectory
        ├── images/                 # Collected RGB image frames
        │   ├── camera_0000.png
        │   └── ...
        ├── groundtruth.txt         # Pose ground truth (T, x, y, z, qw, qx, qy, qz)
        └── images.txt              # Image index file (path timestamp)
```

## 📝 Script Details

*   **`generate_landing_path.py`**: The core logic is the `generate_trajectory_with_orientation` function, which generates smooth landing paths using interpolation and noise superposition.
*   **`collect_from_path.py`**: Uses `airsim.VehicleClient` to control the drone, sets position via `simSetVehiclePose`, and captures images using `simGetImages`.

## ⚠️ Notes
*   Please ensure AirSim's `settings.json` is correctly configured to allow API control.
*   During collection, the script forces the drone position (`ignore_collision=True`). Ensure there are no impassable obstacles blocking the predefined path in the scene.
