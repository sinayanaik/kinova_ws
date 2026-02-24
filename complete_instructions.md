# Refined Prompt for Local LLM Agent

---

## PROJECT: Kinova Gen3 Lite Pick-and-Place — ROS2 Jazzy + Gazebo

**Goal:** Build a color-based cube sorting system using a Kinova Gen3 Lite robot arm in Gazebo simulation. The robot must visually identify colored cubes, physically grasp them, and place them into designated containers. No teleportation, no attachment hacks — real grasp-and-place physics only.

**Stack:** ROS2 Jazzy, Gazebo (gz-sim / Ignition Fortress or Harmonic — whichever pairs with Jazzy), Python 3, Pinocchio (via `pip install pin`), OpenCV, cv_bridge.

**Guiding Principles:**
- Simple, readable, well-documented code with docstrings and inline comments
- Minimal dependencies — only what is strictly necessary
- Exactly 3 packages, no more
- Every file must have a header comment explaining its purpose

---

## WORKSPACE STRUCTURE

```
kinova_ws/
└── src/
    ├── kinova_description/   # URDF, meshes, world, launch
    ├── kinova_control/       # IK, motion planning, state machine
    └── kinova_perception/    # Camera processing, color detection, coordinate estimation
```

---

## PACKAGE 1: `kinova_description`

### 1.1 Robot Model
- Fetch the **official** Kinova Gen3 Lite URDF/xacro, meshes (STL/DAE) from:
  - `https://github.com/Kinovarobotics/ros2_kortex` (primary, ROS2 branch)
  - Fall back to `https://github.com/Kinovarobotics/ros_kortex` and adapt if needed
- Use the **6-DOF Gen3 Lite** variant with the **Robotiq 2F-85 gripper** (or the default Gen3 Lite gripper if 2F-85 is unavailable)
- Ensure the URDF includes:
  - All joint limits (position, velocity, effort)
  - Correct collision and visual geometries
  - Proper inertial properties

### 1.2 Environment (Single Gazebo World File)

**Table:**
- Flat rectangular surface, approx 0.8m × 0.6m, height 0.75m
- Robot base mounted flush on one end of the table (not center — leave working space in front)
- Static model, no physics needed for the table itself

**Cubes (3 total):**
- Size: 0.04m × 0.04m × 0.04m
- Colors: **Red** (1,0,0), **Green** (0,1,0), **Yellow** (1,1,0)
- Placed directly on the table surface in a loose cluster in front of the robot
- Must have proper mass (~0.05 kg), inertia, friction (mu >= 0.8) for realistic grasping
- SDF/URDF models with collision enabled

**Containers (3 total):**
- Open-top box shape: ~0.08m × 0.08m × 0.04m (LxWxH), wall thickness ~0.005m
- Container color mapping (intentionally mismatched to cube colors):
  - Red cube → **Blue** container
  - Green cube → **Orange** container  
  - Yellow cube → **Purple** container
- Placement constraints:
  - All containers within robot reachable workspace (radius ~0.5m from base)
  - Arranged in a semicircular arc or line on the opposite side from the cubes
  - Minimum 0.12m gap between container edges — no overlapping
  - Containers are static (fixed to table)
- Publish the cube-to-container color mapping as a ROS2 parameter or in a YAML config file

### 1.3 Cameras (Gazebo Sensor Plugins)

**Overhead Camera (Top-Down):**
- Position: directly above table center, height ~1.5m above table
- Orientation: pointing straight down (-Z)
- Purpose: sees entire XY workspace, identifies cube positions and colors
- Resolution: 640×480, update rate: 30Hz
- Publishes to: `/camera_overhead/image_raw` (sensor_msgs/Image)

**Side Camera (Lateral):**
- Position: to one side of the table, ~1.0m away, height ~0.5m above table
- Orientation: pointing horizontally toward workspace center
- Purpose: monitors Z-axis (grasp height verification, drop detection)
- Resolution: 640×480, update rate: 30Hz
- Publishes to: `/camera_side/image_raw` (sensor_msgs/Image)

Both cameras must be far enough to never be occluded by the robot arm during normal operation.

### 1.4 Launch File
- Single launch file: `bringup.launch.py`
- Spawns: Gazebo world → robot → cubes → containers → cameras
- Loads robot controllers (joint trajectory controller + gripper controller via `ros2_control`)
- Arguments for: `use_sim_time:=true`, world file path, initial cube positions

### 1.5 ros2_control Integration
- Hardware interface: `gz_ros2_control` (Gazebo plugin)
- Controllers to load:
  - `joint_trajectory_controller` for the 6 arm joints
  - `gripper_action_controller` (or position controller) for the gripper fingers
- Provide a `controllers.yaml` config with all joint names, PID gains, and command interfaces

---

## PACKAGE 2: `kinova_perception`

### 2.1 Node: `color_detector_node`

**Subscriptions:**
- `/camera_overhead/image_raw`
- `/camera_side/image_raw`

**Publications:**
- `/camera_overhead/processed` — raw annotated image with bounding boxes and labels
- `/camera_side/processed` — raw annotated image with bounding boxes and labels
- `/perception/cube_detections` — custom message or `geometry_msgs/PoseArray` with color labels
- `/perception/ee_position` — estimated end-effector position from vision (cross-reference)

### 2.2 Detection Algorithm

**Color Segmentation Pipeline (per frame):**
1. Convert BGR → HSV
2. Define HSV ranges for red, green, yellow (with tunable parameters via ROS2 params)
3. For each color: threshold → morphological open/close (5×5 kernel) → find contours
4. Filter contours by area (min 200px, max 5000px — tunable) to reject noise
5. Compute centroid of each valid contour
6. Use known camera intrinsics + extrinsics (from URDF/TF) to back-project 2D centroid to 3D world coordinates
   - Overhead camera: known Z (table height) → solve for X,Y
   - Side camera: known depth axis → solve for Y,Z (or X,Z depending on orientation)
7. Fuse both camera estimates: overhead gives (X,Y), side gives height (Z) → combined 3D pose
8. Annotate frames with bounding boxes, color labels, and 3D coordinate text

**End-Effector Tracking:**
- Read end-effector pose from TF tree (`base_link` → `end_effector_link`)
- Optionally cross-validate with visual detection (the gripper will have a distinct shape)

### 2.3 Configuration
- `perception_params.yaml`:
  - HSV ranges per color (H_min, H_max, S_min, S_max, V_min, V_max)
  - Camera intrinsic parameters (if not from camera_info topic)
  - Contour area thresholds
  - Detection confidence threshold

---

## PACKAGE 3: `kinova_control`

### 3.1 Inverse Kinematics with Pinocchio

**Setup:**
- Load URDF into Pinocchio model at node startup
- Build the kinematic model and data objects
- Define the end-effector frame ID from the model

**IK Solver — Damped Least Squares (iterative):**
```
Algorithm: IK_solve(target_pose)
  q = current_joint_positions
  for i in 1..max_iterations (200):
      fk = forward_kinematics(q)
      error = pose_difference(target_pose, fk)  # 6D: [position_error; orientation_error_as_log3]
      if norm(error) < tolerance (1e-3):
          return q
      J = compute_jacobian(q)  # 6×6 Jacobian
      # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
      lambda = 1e-6
      dq = J.T @ inv(J @ J.T + lambda * I) @ error
      q = q + alpha * dq   # alpha = step size ~1.0
      q = clamp(q, joint_lower_limits, joint_upper_limits)
  return FAILURE
```

### 3.2 Waypoint Definitions

Define in a YAML config (`waypoints.yaml`) — all values in meters, relative to robot base frame:

| Waypoint Name | Purpose | Position (x, y, z) | Notes |
|---|---|---|---|
| `home` | Neutral upright pose | (0.0, 0.0, 0.45) | Elbow up, safe resting position |
| `observe` | Above cube cluster | (0.35, 0.0, 0.35) | Start of pick sequence, camera-friendly |
| `pre_grasp` | 0.08m above target cube | (cube_x, cube_y, cube_z + 0.08) | Computed dynamically per cube |
| `grasp` | At cube surface | (cube_x, cube_y, cube_z + 0.02) | Gripper open, descend to cube |
| `post_grasp` | Lift after grasp | (cube_x, cube_y, cube_z + 0.12) | Verify grasp here before moving |
| `transit` | Safe travel height | (current_x, current_y, 0.35) | Always lift to this Z before lateral moves |
| `pre_place` | Above target container | (container_x, container_y, 0.25) | Computed per container |
| `place` | Inside container | (container_x, container_y, container_z + 0.06) | Release cube here |
| `post_place` | Retreat after release | (container_x, container_y, 0.25) | Move up before going to next target |

### 3.3 Motion Execution

**Trajectory Generation:**
- Between waypoints: linear interpolation in Cartesian space (10-20 intermediate points)
- At each interpolated point: solve IK → get joint angles
- Package as `trajectory_msgs/JointTrajectory` with time-from-start for each point
- Send via `FollowJointTrajectory` action client to the `joint_trajectory_controller`
- Velocity: conservative — 2-3 seconds per waypoint transition

**Gripper Control:**
- Open command: send finger position = 0.0 (fully open)
- Close command: send finger position = 0.8 (mostly closed — tuned for 0.04m cube)
- Wait 1.0 second after gripper command before next motion

### 3.4 State Machine (Main Orchestrator)

**Node:** `pick_and_place_node`

```
States:
  INITIALIZE
    → Load waypoints, initialize IK solver, wait for controllers
    → Move to HOME
    → Transition → OBSERVE

  OBSERVE
    → Move to OBSERVE waypoint
    → Subscribe to /perception/cube_detections
    → Wait until at least one cube detected with stable position (3 consecutive readings within 0.01m)
    → Select next cube to pick (nearest first, or any unpicked)
    → Store cube_color, cube_position, target_container_position
    → Transition → PRE_GRASP

  PRE_GRASP
    → Open gripper fully
    → Compute pre_grasp waypoint from cube_position
    → Move: current → TRANSIT height → PRE_GRASP
    → Transition → GRASP

  GRASP
    → Descend: PRE_GRASP → GRASP waypoint (slow, 3 sec)
    → Close gripper
    → Wait 1.0 sec
    → Transition → VERIFY_GRASP

  VERIFY_GRASP
    → Move to POST_GRASP (lift 0.12m)
    → Check grasp success:
        Method 1: Read gripper finger positions — if fingers closed beyond cube width (< 0.035m gap), cube likely NOT grasped
        Method 2: Query /perception — is the cube still detected at its original table position?
        Method 3: Check gripper effort/force if available from ros2_control
    → Use at least TWO of the above methods
    → If grasp FAILED:
        → Open gripper
        → Return to OBSERVE (re-detect cube position, it may have shifted)
    → If grasp SUCCEEDED:
        → Transition → TRANSIT_TO_PLACE

  TRANSIT_TO_PLACE
    → Move: POST_GRASP → TRANSIT height → PRE_PLACE (above target container)
    → During transit: periodically check (every 0.5 sec) if cube still grasped
        → Read gripper state
        → If cube dropped:
            → Record approximate drop position (last known EE position)
            → Transition → RECOVER_DROP
    → If transit complete with cube held:
        → Transition → PLACE

  PLACE
    → Descend: PRE_PLACE → PLACE waypoint (inside container)
    → Open gripper
    → Wait 0.5 sec
    → Move to POST_PLACE
    → Transition → VERIFY_PLACE

  VERIFY_PLACE
    → Query /perception — is the cube detected inside the container region?
    → If YES: mark cube as sorted, Transition → OBSERVE (for next cube)
    → If NO or cube detected elsewhere: Transition → OBSERVE (re-pick that cube)
    → If all 3 cubes sorted: Transition → COMPLETE

  RECOVER_DROP
    → Move to HOME (safe position)
    → Move to OBSERVE
    → Wait for perception to re-detect the dropped cube
    → Transition → PRE_GRASP with updated cube position

  COMPLETE
    → Move to HOME
    → Log success
    → Shut down or idle
```

### 3.5 Grasp Verification Detail

```
Algorithm: verify_grasp()
  checks_passed = 0
  total_checks = 2

  # Check 1: Gripper finger gap
  finger_pos = read_gripper_position()  # 0.0=open, 1.0=fully closed
  expected_closed_on_cube = 0.4 to 0.7 range (fingers stopped by cube)
  if finger_pos in expected range:
      checks_passed += 1

  # Check 2: Visual verification
  cube_detections = get_latest_from(/perception/cube_detections)
  cube_still_on_table = is_cube_at_original_position(target_color, original_pos, tolerance=0.03m)
  if NOT cube_still_on_table:
      checks_passed += 1  # cube gone from table = likely in gripper

  return checks_passed >= 2
```

### 3.6 Mid-Transit Drop Detection

```
Algorithm: monitor_during_transit()
  while robot is moving between waypoints:
      every 0.5 seconds:
          finger_pos = read_gripper_position()
          if finger_pos < 0.3 or finger_pos > 0.85:
              # Fingers too open or fully closed (cube slipped out)
              abort_current_trajectory()
              return DROP_DETECTED
  return TRANSIT_COMPLETE
```

---

## FILE MANIFEST

```
kinova_ws/src/
├── kinova_description/
│   ├── CMakeLists.txt
│   ├── package.xml
│   ├── urdf/
│   │   └── gen3_lite_macro.xacro        # Main robot xacro (from Kinova repo)
│   │   └── gen3_lite_environment.urdf.xacro  # Full scene: robot + table + cubes + containers + cameras
│   ├── meshes/                           # All STL/DAE from Kinova repo
│   ├── config/
│   │   ├── controllers.yaml              # ros2_control controller config
│   │   ├── waypoints.yaml                # Named waypoints
│   │   └── cube_container_mapping.yaml   # red→blue, green→orange, yellow→purple
│   ├── worlds/
│   │   └── pick_and_place.sdf            # Gazebo world (lighting, ground plane, physics)
│   └── launch/
│       └── bringup.launch.py             # Master launch file
│
├── kinova_perception/
│   ├── CMakeLists.txt (or setup.py for ament_python)
│   ├── package.xml
│   ├── config/
│   │   └── perception_params.yaml
│   ├── kinova_perception/
│   │   ├── __init__.py
│   │   └── color_detector_node.py
│   └── launch/
│       └── perception.launch.py
│
└── kinova_control/
    ├── CMakeLists.txt (or setup.py for ament_python)
    ├── package.xml
    ├── config/
    │   └── control_params.yaml
    ├── kinova_control/
    │   ├── __init__.py
    │   ├── ik_solver.py                  # Pinocchio IK wrapper
    │   ├── motion_executor.py            # Trajectory generation and execution
    │   ├── gripper_controller.py         # Gripper open/close commands
    │   ├── grasp_verifier.py             # Grasp success checking
    │   └── pick_and_place_node.py        # Main state machine
    └── launch/
        └── control.launch.py
```

---

## BUILD AND RUN COMMANDS

```bash
# Build
cd ~/kinova_ws
colcon build --symlink-install
source install/setup.bash

# Launch everything
ros2 launch kinova_description bringup.launch.py
```
also have a launch script 
launch.sh : this should source the workspace etc that are needed for the launch

---

## CRITICAL CONSTRAINTS — DO NOT VIOLATE

1. **No attach/detach plugins** — The cube must be held by actual gripper friction. Use high friction coefficients (mu1=mu2=1.0) on both gripper fingers and cubes.
2. **No teleporting objects** — All cube movement must be through physical contact with the gripper.
3. **Gripper must physically close on the cube** — Verify finger positions actually correspond to holding something.
4. **All perception must come from the two cameras** — No reading ground-truth poses from Gazebo's model state topic for decision-making. (You may use ground-truth only for debug logging.)
5. **All IK must use Pinocchio** — No MoveIt, no KDL, no analytical shortcuts.
6. **Containers must be within reachable workspace** — Verify with forward kinematics at init time.
7. **Every state transition must be logged** — `self.get_logger().info(...)` with timestamp, current state, reason for transition.
8. **one launch script should run everything: ros2 launch kinova_description bringup.launch.py**