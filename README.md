# Kinova Gen3 Lite Pick-and-Place (Gazebo / ROS2 Jazzy)

A full simulation stack for sorting colored cubes with a Kinova Gen3 Lite 6-DOF arm running inside Gazebo Harmonic. The arm sees a table with six small cubes — two red, two green, two yellow — and moves each one into its matching container, using a custom Pinocchio-based IK solver rather than MoveIt. Everything runs under ROS2 Jazzy, builds with colcon, and launches from a single command.

---

## What the robot actually does

Six 3.5 cm cubes sit in a single row on a table. Three colored containers wait on the far side. An overhead camera looks straight down at the table. The robot homes itself, looks at what the camera sees, picks the next cube in a predetermined order (red → green → yellow → red → green → yellow), carries it over, and drops it into the right container. It repeats until all six are sorted, then parks back at home.

The whole sequence is a nine-state machine: INITIALIZE → OBSERVE → PRE\_GRASP → GRASP → VERIFY\_GRASP → TRANSIT\_TO\_PLACE → PLACE → VERIFY\_PLACE → COMPLETE, with RECOVER\_DROP available as a fallback if the arm drops something in transit.

---

## Repository layout

```
kinova_antigravity/
├── launch.sh                          # one-shot build + launch script
└── src/
    ├── kinova_description/            # robot model, world, controllers, launch
    ├── kinova_control/                # IK solver, motion, gripper, state machine
    ├── kinova_perception/             # color detection, 3D pose estimation
    └── kortex_description/            # upstream Kinova URDF/meshes (vendor)
```

---

## Packages

### `kinova_description`

This is the load-bearing CMake package that everything else depends on at launch time. It holds the robot URDF, the Gazebo world, the controller config, and the master bringup launch file. It does not contain Python logic — it is configuration and description only.

| File / folder | Purpose |
|---|---|
| `urdf/gen3_lite_environment.urdf.xacro` | Xacro macro that assembles the Gen3 Lite arm with fingers, mounts an overhead camera, and attaches the `gz_ros2_control` hardware plugin for Gazebo |
| `worlds/pick_and_place.sdf` | Gazebo world file — table, six colored cubes at fixed positions, three containers, lighting, ground plane, and the overhead camera sensor |
| `config/controllers.yaml` | Defines three ros2\_control controllers: `joint_state_broadcaster` (500 Hz), `joint_trajectory_controller` for the six arm joints, and `position_controllers/GripperActionController` for the single gripper joint |
| `config/cube_container_mapping.yaml` | Ground-truth positions for every cube and container in world frame. Red/green/yellow containers sit at x=0.40; cubes sit in a row at x=0.18, spaced 8 cm apart |
| `config/waypoints.yaml` | Named joint configurations (home, observe) and numeric offsets (transit height, pre-grasp height, drop height) used by the state machine |
| `launch/bringup.launch.py` | Master launch file. Starts Gazebo, spawns the robot, chains controller spawners (JSB → arm → gripper, each waiting on the previous to exit), bridges the overhead camera topic via `ros_gz_bridge`, then starts perception and control nodes |

The controller spawning is deliberately chained with `OnProcessExit` event handlers so the trajectory controller never tries to claim joints before the joint state broadcaster has registered them.

---

### `kinova_control`

The Python package that holds the brain of the system. Five modules, one launch file, two config files.

#### `kinova_control/ik_solver.py`

Everything IK-related lives here. Uses [Pinocchio](https://github.com/stack-of-tasks/pinocchio) to load the URDF at runtime, locate the `tool_frame` end-effector frame, and solve inverse kinematics numerically. No MoveIt, no KDL.

**The algorithm is damped least squares, also called Levenberg-Marquardt:**

At each iteration the solver computes the 6D error between the current tool frame placement and the target (position error + orientation error expressed as a Lie algebra element via `pin.log6`). It then computes the frame Jacobian J (6×N) and solves for a joint update:

$$\Delta q = J^T (J J^T + \lambda I)^{-1} e$$

where λ is an adaptive damping factor that scales with error magnitude. The update is integrated on the manifold via `pin.integrate` (which respects the SO(3) structure of joint space) and each joint is clamped to its URDF limits after every step. The solver runs up to 300 iterations per call.

Three solver variants are provided and stacked into a `solve_robust` cascade:

- **Full 6D** — minimises both position error (3 DOF) and orientation error (3 DOF, via SE(3) log map). Most precise.
- **Z-axis-down 5D** — minimises position error and a cross-product Z-axis alignment error, but leaves wrist roll free. Succeeds where full 6D gets stuck in local minima.
- **Position-only 3D** — uses only the top three rows of the Jacobian. Last resort, no orientation constraint at all.

`solve_robust` tries these in order, and each is attempted with: the current joint configuration first, then 15 hand-picked seed configurations spread across the reachable workspace, and finally random restarts for position-only. This makes IK essentially never fail for positions that are geometrically reachable.

A separate method `solve_z_axis_down` specifically enforces that the tool Z-axis points straight down (world −Z) without caring about wrist roll — this is the right constraint for top-down grasping.

**Key design choice:** IK targets `tool_frame` (the actual fingertip frame), not `end_effector_link`. This means the numbers passed to `solve()` are exactly where you want the fingers to be, with no TCP offset arithmetic in the state machine.

#### `kinova_control/motion_executor.py`

Handles the gap between "I want the gripper here" and "a JointTrajectory message exists that lets me get there." Given a target Cartesian position, it:

1. Calls `ik_solver.forward_kinematics` on the current joints to find where the gripper is now.
2. Linearly interpolates N points between the current position and the target (default: 8 steps for standard moves, more for longer distances).
3. Solves IK at each interpolated point, warm-starting from the solution to the previous point so the joint path stays smooth.
4. Falls back to `solve_z_axis_down` at any waypoint where full 6D IK fails.
5. Packages everything into a `JointTrajectory` and sends it to the `FollowJointTrajectory` action server.

The multi-waypoint variant chains multiple Cartesian segments into a single trajectory, which avoids the pause that would result from sending them one at a time.

#### `kinova_control/pick_and_place_node.py`

The top-level ROS2 node and the state machine that drives everything. It subscribes to `/joint_states` (for current arm and finger positions) and `/perception/cube_detections` (for visible cube locations), runs three action clients (arm trajectory, gripper), and drives the state machine from a background thread leaving the ROS spin thread free for callbacks.

**The pick sequence:**

The order is hardcoded as `[red, green, yellow, red, green, yellow]` because the cube layout is fixed. At OBSERVE the node consults the latest `PoseArray` from perception, picks the closest unsorted cube of the current color, notes its world position, looks up the target container from `cube_container_mapping.yaml`, and advances to PRE\_GRASP.

PRE\_GRASP opens the gripper and moves the fingertip to 5 cm above the cube. Before calling IK it runs `_set_desired_wrist_angle` — this checks for neighboring cubes within 12 cm and picks a wrist roll angle (j6) that orients the finger opening direction perpendicular to the nearest neighbor, minimizing the chance of knocking it over. The chosen angle is then applied as a post-IK correction to j6 on every IK call during approach, so the orientation stays locked while the arm descends.

GRASP is two-phase: a coarse descent to 1 cm above the cube (loose IK tolerance), a brief settle, then a slow careful Cartesian descent to 1.5 cm below the cube top for side contact. After closing the gripper it waits 2 seconds for force to build, then does an immediate finger-gap check before even trying to lift — if the fingers are fully closed (nothing between them) it retries rather than carrying an empty gripper all the way to the container.

VERIFY\_GRASP lifts the arm 8 cm and rechecks the finger gap. Finger position in the range [0.15, 0.75] means a cube is present. Only one check is required (the visual disappearance check is optional).

TRANSIT moves to a position 7 cm above the target container using a single `_move_to_pose` call. Drop detection checks the finger gap mid-transit.

PLACE simply opens the gripper and lets the cube fall. The arm retreats upward immediately after.

VERIFY\_PLACE increments the sorted count, advances the pick index, and loops back to OBSERVE.

RECOVER\_DROP opens the gripper, returns to home, and goes back to OBSERVE to re-acquire a target.

The `_move_joints` method does its own 5-waypoint joint-space interpolation between current pose and target, keeping trajectory durations short (0.4–1.5 s) to make the whole sequence fast.

#### `kinova_control/gripper_controller.py`

Thin wrapper around the `GripperCommand` action client. Takes a position (0.0 = fully open, 0.82 = gripping a ~4 cm cube) and effort, sends the goal, awaits the result, and waits an extra settling period. Commands only `right_finger_bottom_joint` — the hardware plugin handles the three mimic joints automatically.

#### `kinova_control/grasp_verifier.py`

Two verification methods:

1. **Finger gap check** — right_finger_bottom_joint position between 0.15 and 0.75 rad means something is in the gripper. Below 0.15 = nothing there. Above 0.75 = cube slipped through.
2. **Visual check** — looks at the latest `PoseArray` from perception and returns `True` if no cube of the target color is detected at the original pick position (within 5 cm tolerance). This is optional and only required if `required_checks: 2` is set.

`DropDetector` uses the same finger position logic during transit, returning `True` if the position drops below 0.1 (fingers suddenly opened) or exceeds 0.84 (cube squeezed out).

#### Config files

`config/control_params.yaml` — all parameters for the node: IK iterations/tolerance, motion durations, gripper open/close positions, grasp/drop thresholds, joint names, Cartesian offsets. These feed into the node's ROS2 parameter server so anything can be tuned without touching Python.

`config/grasp_params.yaml` — a separate param file with just grasp and gripper parameters using the wildcard (`/**`) namespace so it can be loaded standalone.

#### `launch/control.launch.py`

Starts just the pick-and-place node with `control_params.yaml` and a freshly evaluated `robot_description` (xacro → URDF string) injected as a parameter so Pinocchio can build the kinematic model at node startup.

---

### `kinova_perception`

Single-node package. Subscribes to an overhead camera, segments colors, and publishes 3D cube positions.

#### `kinova_perception/color_detector_node.py`

The node does three things: color segmentation, 3D back-projection, and stability filtering.

**Color segmentation** converts each incoming frame from BGR to HSV and applies `cv2.inRange` masks for each color. Red is handled with two ranges (H: 0–10 and 170–180) because red wraps around the hue wheel. After masking, morphological open+close operations clean up noise, then `cv2.findContours` extracts candidate blobs. Contours outside the area range [200, 5000] pixels are discarded. All surviving contours return their centroid and area — not just the largest one, since two cubes of the same color can be visible simultaneously.

**Back-projection** converts 2D pixel centroids to 3D world coordinates. The overhead camera points straight down with a 90° rotation around the Y-axis, which means:

- pixel +u (rightward) maps to world −Y
- pixel +v (downward) maps to world −X

Given that the camera sits at a known world position $(x_c, y_c, z_c)$ and the table surface is at a known Z, the depth is just $z_c - z_{table}$, so the full projection is:

$$x_w = x_c - n_y \cdot d, \quad y_w = y_c - n_x \cdot d$$

where $n_x = (u - c_x)/f_x$, $n_y = (v - c_y)/f_y$, and $d = z_c - (z_{table} + z_{half\_cube})$.

No stereo, no depth sensor, no point cloud. The known table height makes it exact.

**Spatial filtering** rejects any detection with $x < 0.10$ or $x > 0.25$ or $|y| > 0.25$. This is the cube pick zone — the containers sit at $x = 0.40$ and would otherwise appear as false detections.

**Color encoding in PoseArray** — since `PoseArray` carries poses and not arbitrary metadata, color identity is packed into the orientation quaternion: red → (1,0,0,0), green → (0,1,0,0), yellow → (0,0,1,0). The control node decodes this with simple threshold checks on the quaternion components.

**Stability filtering** keeps a rolling history of the last N detected positions per color index. A position is considered stable when the spread across the last 3 readings is less than 1.5 cm. This prevents the arm from chasing a jittering centroid estimate.

On the first 10 frames the node also logs a coordinate verification table comparing detected positions against the known ground-truth positions from `cube_container_mapping.yaml`, which makes it easy to spot a camera calibration mismatch.

#### `config/perception_params.yaml`

HSV band limits, contour area bounds, camera intrinsics (fx, fy, cx, cy for a 640×480 image), camera extrinsic position in the world frame, table height, and stability filter settings. All of these feed into ROS2 parameters so they can be changed for a physical setup without recompiling.

#### `launch/perception.launch.py`

Launches the `color_detector_node` with `perception_params.yaml`.

---

### `kortex_description`

Upstream vendor package from Kinova. Contains the URDF macros, mesh files (DAE/STL), and SRDF for the Gen3 Lite arm and its 2-finger gripper. Nothing in this repo edits these files. The `kinova_description` xacro file includes these macros to build the full robot description.

---

## How the full stack connects

```
Gazebo Harmonic
  │
  ├── publishes /joint_states  ──────────────────────────────► pick_and_place_node
  ├── publishes /camera_overhead/image_raw ──► color_detector_node
  │                                                  │
  │                                           publishes /perception/cube_detections
  │                                                  │
  │                                                  ▼
  │                                           pick_and_place_node
  │                                                  │
  │                                    sends FollowJointTrajectory goals
  │                                                  │
  ◄────────────────── /joint_trajectory_controller ──┘
  ◄────────────────── /gripper_controller ───────────┘
```

The pick-and-place node is the only node that actually commands the robot. Perception only publishes detections — it never sends a motion command. The IK solver runs entirely inside the control node process; there is no separate IK service.

---

## Dependencies

| Dependency | Why |
|---|---|
| ROS2 Jazzy | Base middleware |
| Gazebo Harmonic | Simulation environment |
| `gz_ros2_control` / `ros2_control` | Hardware abstraction and trajectory controllers |
| `ros_gz_bridge` | Bridges Gazebo camera topics into ROS2 |
| `robot_state_publisher` + `xacro` | URDF processing and TF tree |
| `pinocchio` | Rigid body kinematics and IK (Python bindings) |
| `opencv-python` + `cv_bridge` | Color segmentation and ROS↔OpenCV conversion |
| `numpy` | All numerical operations in IK and perception |

---

## Building and launching

```bash
# Full build + launch (first time or after source changes)
./launch.sh

# Manual build
source /opt/ros/jazzy/setup.bash
cd kinova_antigravity
colcon build --symlink-install
source install/setup.bash

# Launch individual components
ros2 launch kinova_description bringup.launch.py

# Perception only
ros2 launch kinova_perception perception.launch.py

# Control only (assumes Gazebo + controllers already running)
ros2 launch kinova_control control.launch.py
```

`launch.sh` kills any leftover Gazebo and ROS processes first, sources Jazzy, sources the workspace install overlay, and runs `bringup.launch.py`. If the workspace hasn't been built yet it triggers `colcon build` automatically.

---

## Configuration quick reference

| File | What to change there |
|---|---|
| `kinova_description/config/cube_container_mapping.yaml` | Cube and container world positions, number of cubes per color |
| `kinova_description/config/waypoints.yaml` | Named joint configurations, transit height, grasp/place offsets |
| `kinova_description/config/controllers.yaml` | Controller update rates, joint names, velocity tolerance |
| `kinova_control/config/control_params.yaml` | IK iterations, motion duration, gripper positions, grasp thresholds |
| `kinova_perception/config/perception_params.yaml` | HSV color bands, camera intrinsics+extrinsics, stability filter |

---

## Notes on IK robustness

The solver's 15 seed configurations were found by scanning the Gen3 Lite's workspace with forward kinematics — specifically, configurations where the tool frame Z-axis points down and the fingertip is somewhere over the table. Without good seeds, a gradient-descent IK on a 6-DOF arm with tight joint limits will frequently converge to a local minimum that satisfies the equations but places the tool in completely the wrong part of the workspace. The seed bank plus the 6D → 5D → 3D cascade makes IK failures essentially impossible for any table-height position within about 40 cm of the arm base.

The wrist alignment logic (finger orientation avoiding neighbors) uses j6 — the last joint — to rotate the gripper around the vertical axis after IK is solved. Because j6 only affects rotation around the tool Z-axis, adjusting it doesn't change the fingertip position or the vertical orientation, so the IK solution stays valid after the correction.
