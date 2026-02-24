#!/usr/bin/env python3
"""
pick_and_place_node.py
Main state machine orchestrator for the Kinova Gen3 Lite pick-and-place.

Implements the complete pick-and-place cycle:
  INITIALIZE → OBSERVE → PRE_GRASP → GRASP → VERIFY_GRASP →
  TRANSIT_TO_PLACE → PLACE → VERIFY_PLACE → COMPLETE
  (with RECOVER_DROP fallback)

Uses a background thread for the state machine so that ROS2 callbacks
(joint states, detections) continue processing on the main executor.
All waits use time.sleep(), not asyncio.
"""
import time
import threading
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from rclpy.action import ActionClient
from ament_index_python.packages import get_package_share_directory

from kinova_control.ik_solver import IKSolver
from kinova_control.gripper_controller import GripperController
from kinova_control.grasp_verifier import GraspVerifier, DropDetector


class PickAndPlaceState:
    """Enumeration of all state machine states."""
    INITIALIZE = 'INITIALIZE'
    OBSERVE = 'OBSERVE'
    PRE_GRASP = 'PRE_GRASP'
    GRASP = 'GRASP'
    VERIFY_GRASP = 'VERIFY_GRASP'
    TRANSIT_TO_PLACE = 'TRANSIT_TO_PLACE'
    PLACE = 'PLACE'
    VERIFY_PLACE = 'VERIFY_PLACE'
    RECOVER_DROP = 'RECOVER_DROP'
    COMPLETE = 'COMPLETE'


class PickAndPlaceNode(Node):
    """
    ROS2 node implementing the complete pick-and-place state machine.

    The state machine runs in a background thread so that ROS2 callbacks
    (joint states, perception) continue to be processed by the main executor.
    """

    def __init__(self):
        super().__init__('pick_and_place_node')
        self.get_logger().info('=' * 60)
        self.get_logger().info('Initializing Pick-and-Place State Machine...')
        self.get_logger().info('=' * 60)

        # Callback group for async operations
        self.cb_group = ReentrantCallbackGroup()

        # ---- Declare parameters ----
        self._declare_all_parameters()

        # ---- State machine ----
        self.state = PickAndPlaceState.INITIALIZE
        self.cubes_sorted = {}  # color -> count of cubes sorted
        self.total_cubes = 6  # 2 per color
        self.current_cube_color = None
        self.current_cube_pos = None
        self.current_container_pos = None
        self.retry_count = 0
        self.max_retries = 3

        # ---- Latest sensor data ----
        self.latest_joint_state = None
        self.latest_cube_detections = None
        self.arm_joint_names = None
        self.gripper_joint_names = None
        self.current_arm_positions = np.zeros(6)
        self.current_finger_pos = 0.0

        # ---- Load config files ----
        self._load_configs()

        # ---- Subscribers ----
        self.joint_state_sub = self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_callback, 10,
            callback_group=self.cb_group
        )
        self.detections_sub = self.create_subscription(
            PoseArray, '/perception/cube_detections',
            self._detections_callback, 10,
            callback_group=self.cb_group
        )

        # ---- Robot description (for Pinocchio) ----
        self.declare_parameter('robot_description', '')
        self.ik_solver = None
        self.gripper_controller = None
        self.grasp_verifier = None
        self.drop_detector = None

        # ---- Arm action client ----
        self.arm_action_client = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory'
        )

        # ---- Grasp orientation: gripper pointing straight down ----
        self.grasp_orientation = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])

        # ---- Start the state machine in a background thread after delay ----
        self._sm_thread = threading.Thread(target=self._state_machine_thread, daemon=True)
        self._sm_thread.start()

        self.get_logger().info('Node created. State machine thread started.')

    def _declare_all_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        # IK
        self.declare_parameter('ik.max_iterations', 200)
        self.declare_parameter('ik.tolerance', 0.001)
        self.declare_parameter('ik.damping_lambda', 1e-6)
        self.declare_parameter('ik.step_size', 1.0)

        # Motion
        self.declare_parameter('motion.interpolation_steps', 15)
        self.declare_parameter('motion.waypoint_duration', 2.5)
        self.declare_parameter('motion.slow_duration', 3.0)
        self.declare_parameter('motion.transit_height', 0.35)

        # Gripper
        self.declare_parameter('gripper.open_position', 0.0)
        self.declare_parameter('gripper.close_position', 0.8)
        self.declare_parameter('gripper.max_effort', 100.0)
        self.declare_parameter('gripper.wait_after_command', 1.0)

        # Grasp verification
        self.declare_parameter('grasp.finger_min_expected', 0.4)
        self.declare_parameter('grasp.finger_max_expected', 0.75)
        self.declare_parameter('grasp.position_tolerance', 0.03)
        self.declare_parameter('grasp.required_checks', 2)

        # Drop detection
        self.declare_parameter('drop.finger_min_threshold', 0.3)
        self.declare_parameter('drop.finger_max_threshold', 0.85)

        # Offsets
        self.declare_parameter('offsets.pre_grasp_z', 0.08)
        self.declare_parameter('offsets.grasp_z', 0.02)
        self.declare_parameter('offsets.post_grasp_z', 0.12)
        self.declare_parameter('offsets.pre_place_z', 0.25)
        self.declare_parameter('offsets.place_z', 0.06)

        # Joint names
        self.declare_parameter('arm_joint_names',
            ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6'])
        self.declare_parameter('gripper_joint_names',
            ['right_finger_bottom_joint', 'right_finger_tip_joint',
             'left_finger_bottom_joint', 'left_finger_tip_joint'])

    def _load_configs(self):
        """Load cube/container mapping and waypoints from YAML files."""
        desc_share = get_package_share_directory('kinova_description')

        # Load cube-container mapping
        mapping_file = f'{desc_share}/config/cube_container_mapping.yaml'
        try:
            with open(mapping_file, 'r') as f:
                mapping_data = yaml.safe_load(f)
            self.cube_mapping = mapping_data.get('cube_container_mapping', {})
            self.get_logger().info(f'Loaded cube-container mapping: {list(self.cube_mapping.keys())}')
        except Exception as e:
            self.get_logger().error(f'Failed to load cube-container mapping: {e}')
            self.cube_mapping = {}

        # Load waypoints
        waypoints_file = f'{desc_share}/config/waypoints.yaml'
        try:
            with open(waypoints_file, 'r') as f:
                wp_data = yaml.safe_load(f)
            self.waypoints = wp_data.get('waypoints', {})
            self.get_logger().info(f'Loaded waypoints config.')
        except Exception as e:
            self.get_logger().error(f'Failed to load waypoints: {e}')
            self.waypoints = {}

        # Get joint names
        self.arm_joint_names = self.get_parameter('arm_joint_names').value
        self.gripper_joint_names = self.get_parameter('gripper_joint_names').value

    # ==================== CALLBACKS ====================

    def _joint_state_callback(self, msg: JointState):
        """Store latest joint state data."""
        self.latest_joint_state = msg
        try:
            positions = np.zeros(6)
            for i, name in enumerate(self.arm_joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions[i] = msg.position[idx]
            self.current_arm_positions = positions

            if 'right_finger_bottom_joint' in msg.name:
                idx = msg.name.index('right_finger_bottom_joint')
                self.current_finger_pos = msg.position[idx]
        except (ValueError, IndexError):
            pass

    def _detections_callback(self, msg: PoseArray):
        """Store latest cube detections from perception."""
        self.latest_cube_detections = msg

    def _transition(self, new_state: str, reason: str):
        """Transition to a new state with logging."""
        now = self.get_clock().now()
        self.get_logger().info(
            f'[{now.nanoseconds / 1e9:.2f}] STATE TRANSITION: '
            f'{self.state} → {new_state} | Reason: {reason}'
        )
        self.state = new_state

    # ==================== SYNCHRONOUS MOTION HELPERS ====================

    def _send_joint_trajectory_blocking(self, joint_positions_list, duration=3.0):
        """
        Send arm joint positions via FollowJointTrajectory action, blocking until done.

        Args:
            joint_positions_list: List of 6-element arrays (one per waypoint)
            duration: Total trajectory duration in seconds

        Returns:
            True if successful
        """
        if not self.arm_action_client.server_is_ready():
            self.get_logger().error('Arm trajectory action server not ready!')
            return False

        trajectory = JointTrajectory()
        trajectory.joint_names = self.arm_joint_names

        n = len(joint_positions_list)
        dt = duration / max(n, 1)
        for i, positions in enumerate(joint_positions_list):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            # Only set zero velocity at the LAST point (goal); let controller
            # interpolate smoothly between intermediate points.
            if i == n - 1:
                point.velocities = [0.0] * 6
            t = (i + 1) * dt
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9)
            )
            trajectory.points.append(point)

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.get_logger().info(
            f'Sending trajectory: {len(trajectory.points)} points, '
            f'{duration:.1f}s duration'
        )

        # Send goal and wait for acceptance
        send_future = self.arm_action_client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Trajectory goal REJECTED')
            return False

        self.get_logger().info('Trajectory accepted, executing...')

        # Wait for result
        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.05)

        result = result_future.result()
        if result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.get_logger().info('Trajectory completed successfully.')
            return True
        else:
            self.get_logger().warn(
                f'Trajectory finished with code: {result.result.error_code}'
            )
            return True  # Still treat as done

    def _move_to_joint_angles(self, target_angles, duration=4.0):
        """
        Move the arm to specific joint angles.

        Interpolates from current position to target in joint space.
        """
        current = self.current_arm_positions.copy()
        target = np.array(target_angles)

        # Interpolate in joint space (10 intermediate waypoints for smoothness)
        n_steps = 10
        waypoints = []
        for i in range(1, n_steps + 1):
            t = i / float(n_steps)
            wp = current + t * (target - current)
            waypoints.append(wp)

        return self._send_joint_trajectory_blocking(waypoints, duration)

    def _move_to_cartesian(self, target_pos_world, orientation=None, duration=3.0):
        """
        Move end-effector to a Cartesian position in world frame.

        Uses IK with Cartesian interpolation from current EE position.
        """
        if self.ik_solver is None:
            self.get_logger().error('IK solver not initialized!')
            return False

        if orientation is None:
            orientation = self.grasp_orientation

        # Get current EE position from FK
        current_ee = self.ik_solver.forward_kinematics(self.current_arm_positions)
        start_pos = current_ee.translation.copy()

        target_pos = np.array(target_pos_world)

        # Interpolate in Cartesian space
        n_steps = 10
        q_prev = self.current_arm_positions.copy()
        waypoints = []

        for i in range(1, n_steps + 1):
            t = i / float(n_steps)
            wp_pos = start_pos + t * (target_pos - start_pos)

            success, q_sol = self.ik_solver.solve(
                wp_pos, orientation, q_prev,
                max_iterations=300, tolerance=1e-3, damping=1e-6
            )
            if not success:
                success, q_sol = self.ik_solver.solve_position_only(
                    wp_pos, orientation, q_prev,
                    max_iterations=300, tolerance=1e-3
                )
                if not success:
                    self.get_logger().warn(
                        f'IK failed at step {i}/{n_steps}: pos={wp_pos}'
                    )
                    return False

            waypoints.append(q_sol)
            q_prev = q_sol.copy()

        return self._send_joint_trajectory_blocking(waypoints, duration)

    def _send_gripper_blocking(self, position, effort=100.0):
        """Send gripper command and wait for completion."""
        if self.gripper_controller is None:
            self.get_logger().error('Gripper controller not initialized!')
            return False

        from control_msgs.action import GripperCommand

        if not self.gripper_controller.action_client.server_is_ready():
            self.get_logger().error('Gripper action server not ready!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = effort

        self.get_logger().info(f'Gripper command: position={position:.2f}')

        send_future = self.gripper_controller.action_client.send_goal_async(goal)
        while not send_future.done():
            time.sleep(0.05)

        goal_handle = send_future.result()
        if not goal_handle.accepted:
            self.get_logger().error('Gripper goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        while not result_future.done():
            time.sleep(0.05)

        time.sleep(0.5)  # Let gripper settle
        return True

    def _open_gripper(self):
        """Open gripper fully."""
        return self._send_gripper_blocking(0.0)

    def _close_gripper(self):
        """Close gripper for grasping."""
        close_pos = self.get_parameter('gripper.close_position').value
        effort = self.get_parameter('gripper.max_effort').value
        return self._send_gripper_blocking(close_pos, effort)

    # ==================== STATE MACHINE THREAD ====================

    def _state_machine_thread(self):
        """Background thread running the state machine."""
        self.get_logger().info('State machine thread: waiting 15s for Gazebo + controllers...')
        time.sleep(15.0)  # Wait for Gazebo, controllers, sensors to fully initialize

        # Initialize system
        if not self._initialize_system():
            self.get_logger().error('System initialization failed! Exiting SM.')
            return

        # Run state machine loop
        while rclpy.ok():
            try:
                if self.state == PickAndPlaceState.INITIALIZE:
                    self._state_initialize()
                elif self.state == PickAndPlaceState.OBSERVE:
                    self._state_observe()
                elif self.state == PickAndPlaceState.PRE_GRASP:
                    self._state_pre_grasp()
                elif self.state == PickAndPlaceState.GRASP:
                    self._state_grasp()
                elif self.state == PickAndPlaceState.VERIFY_GRASP:
                    self._state_verify_grasp()
                elif self.state == PickAndPlaceState.TRANSIT_TO_PLACE:
                    self._state_transit_to_place()
                elif self.state == PickAndPlaceState.PLACE:
                    self._state_place()
                elif self.state == PickAndPlaceState.VERIFY_PLACE:
                    self._state_verify_place()
                elif self.state == PickAndPlaceState.RECOVER_DROP:
                    self._state_recover_drop()
                elif self.state == PickAndPlaceState.COMPLETE:
                    self._state_complete()
                    break  # Done
                elif self.state == 'IDLE':
                    break  # Done
                else:
                    time.sleep(0.5)
            except Exception as e:
                self.get_logger().error(f'State machine error in {self.state}: {e}')
                import traceback
                self.get_logger().error(traceback.format_exc())
                time.sleep(2.0)

    def _initialize_system(self):
        """Initialize IK solver, gripper controller, etc."""
        self.get_logger().info('Starting system initialization...')

        # Wait for robot_description
        robot_desc = ''
        for attempt in range(30):
            try:
                robot_desc = self.get_parameter('robot_description').value
                if robot_desc:
                    break
            except Exception:
                pass
            self.get_logger().info(f'Waiting for robot_description... ({attempt+1})')
            time.sleep(1.0)

        if not robot_desc:
            self.get_logger().error('No robot_description!')
            return False

        # Initialize IK solver
        try:
            self.ik_solver = IKSolver(robot_desc, 'end_effector_link')
            self.get_logger().info('IK solver initialized.')
        except Exception as e:
            self.get_logger().error(f'IK solver init failed: {e}')
            return False

        # Initialize gripper controller
        self.gripper_controller = GripperController(
            node=self,
            open_position=self.get_parameter('gripper.open_position').value,
            close_position=self.get_parameter('gripper.close_position').value,
            max_effort=self.get_parameter('gripper.max_effort').value,
            wait_after_command=self.get_parameter('gripper.wait_after_command').value,
        )

        # Initialize verifiers
        self.grasp_verifier = GraspVerifier(
            finger_min_expected=self.get_parameter('grasp.finger_min_expected').value,
            finger_max_expected=self.get_parameter('grasp.finger_max_expected').value,
            position_tolerance=self.get_parameter('grasp.position_tolerance').value,
            required_checks=self.get_parameter('grasp.required_checks').value,
        )
        self.drop_detector = DropDetector(
            finger_min_threshold=self.get_parameter('drop.finger_min_threshold').value,
            finger_max_threshold=self.get_parameter('drop.finger_max_threshold').value,
        )

        # Wait for action servers
        self.get_logger().info('Waiting for arm trajectory action server...')
        if not self.arm_action_client.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Arm trajectory action server not available!')
            return False
        self.get_logger().info('Arm action server ready.')

        self.get_logger().info('Waiting for gripper action server...')
        if not self.gripper_controller.wait_for_server(timeout_sec=30.0):
            self.get_logger().error('Gripper action server not available!')
            return False
        self.get_logger().info('Gripper action server ready.')

        # Wait for joint states
        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            if self.latest_joint_state is not None:
                break
            time.sleep(0.2)

        if self.latest_joint_state is None:
            self.get_logger().error('No joint states!')
            return False

        # ---- FK VERIFICATION: log EE positions for all named waypoints ----
        self.get_logger().info('=' * 50)
        self.get_logger().info('FK VERIFICATION of waypoints:')
        for wp_name in ['home', 'observe']:
            wp_data = self.waypoints.get(wp_name, {})
            angles = wp_data.get('joint_angles')
            if angles:
                q = np.array(angles)
                ee_pose = self.ik_solver.forward_kinematics(q)
                pos = ee_pose.translation
                self.get_logger().info(
                    f'  {wp_name}: joints={[f"{a:.3f}" for a in angles]} '
                    f'→ EE position=[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]'
                )
        # Also log current (zero-config) FK
        q_zero = np.zeros(6)
        ee_zero = self.ik_solver.forward_kinematics(q_zero)
        pos_zero = ee_zero.translation
        self.get_logger().info(
            f'  zero-config → EE position=[{pos_zero[0]:.4f}, {pos_zero[1]:.4f}, {pos_zero[2]:.4f}]'
        )
        self.get_logger().info('=' * 50)

        self.get_logger().info('System initialization complete!')
        return True

    # ==================== STATE IMPLEMENTATIONS ====================

    def _state_initialize(self):
        """INITIALIZE: Move to home position using joint angles."""
        home_config = self.waypoints.get('home', {})
        home_angles = home_config.get('joint_angles', [0.0, -0.35, 1.4, 0.0, -1.05, 0.0])

        self.get_logger().info(f'Moving to HOME joint angles: {home_angles}')
        success = self._move_to_joint_angles(home_angles, duration=4.0)
        time.sleep(1.0)

        if success:
            self._transition(PickAndPlaceState.OBSERVE,
                           'Moved to HOME, starting observation')
        else:
            self.get_logger().warn('Failed to move to HOME, retrying in 3s...')
            time.sleep(3.0)

    def _state_observe(self):
        """OBSERVE: Move to observation position and look for cubes."""
        observe_config = self.waypoints.get('observe', {})
        observe_angles = observe_config.get('joint_angles', [0.0, 0.0, 1.57, 0.0, -1.57, 0.0])

        self.get_logger().info(f'Moving to OBSERVE joint angles: {observe_angles}')
        success = self._move_to_joint_angles(observe_angles, duration=3.0)
        if not success:
            self.get_logger().warn('Failed to reach OBSERVE position')
            time.sleep(2.0)
            return

        # Wait for perception to stabilize
        self.get_logger().info('Observing cubes (waiting 3s for perception)...')
        time.sleep(3.0)

        if self.latest_cube_detections is None:
            self.get_logger().warn('No cube detections yet, waiting...')
            time.sleep(2.0)
            return

        # Select next cube
        selected = self._select_next_cube()
        if selected is None:
            total_sorted = sum(self.cubes_sorted.values())
            if total_sorted >= self.total_cubes:
                self._transition(PickAndPlaceState.COMPLETE, f'All {self.total_cubes} cubes sorted')
            else:
                self.get_logger().warn(f'No unpicked cubes detected ({total_sorted}/{self.total_cubes} sorted), waiting...')
                time.sleep(3.0)
            return

        color, cube_pos = selected
        self.current_cube_color = color
        self.current_cube_pos = cube_pos

        # Look up target container
        container_info = self.cube_mapping.get(color, {})
        container_pos = container_info.get('container_position')
        if container_pos:
            self.current_container_pos = np.array(container_pos)
        else:
            self.get_logger().error(f'No container mapping for {color}!')
            return

        self.get_logger().info(
            f'Selected {color} cube at {cube_pos} → '
            f'{container_info.get("target_container")} at {container_pos}'
        )
        self._transition(PickAndPlaceState.PRE_GRASP, f'Selected {color} cube')

    def _select_next_cube(self):
        """Select the nearest unpicked cube. Each color can have multiple cubes."""
        if not self.latest_cube_detections:
            return None

        color_map = {
            'red': lambda q: q.x > 0.5,
            'green': lambda q: q.y > 0.5,
            'yellow': lambda q: q.z > 0.5,
        }

        # Get max cubes per color from mapping
        max_per_color = {}
        for color in ['red', 'green', 'yellow']:
            info = self.cube_mapping.get(color, {})
            max_per_color[color] = info.get('num_cubes', 2)

        candidates = []
        for pose in self.latest_cube_detections.poses:
            for color, check_fn in color_map.items():
                # Skip colors that have all cubes sorted
                sorted_count = self.cubes_sorted.get(color, 0)
                if sorted_count >= max_per_color.get(color, 2):
                    continue
                if check_fn(pose.orientation):
                    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    dist = np.linalg.norm(pos[:2])
                    candidates.append((color, pos, dist))

        if not candidates:
            return None

        candidates.sort(key=lambda c: c[2])
        return candidates[0][0], candidates[0][1]

    def _state_pre_grasp(self):
        """PRE_GRASP: Open gripper and move above the target cube."""
        self._open_gripper()

        # Pre-grasp: above cube in world frame
        pre_grasp_z_offset = self.get_parameter('offsets.pre_grasp_z').value
        pre_grasp_pos = self.current_cube_pos.copy()
        pre_grasp_pos[2] += pre_grasp_z_offset

        self.get_logger().info(f'Moving to PRE_GRASP at {pre_grasp_pos} (world frame)')
        success = self._move_to_cartesian(pre_grasp_pos, duration=3.0)

        if success:
            self._transition(PickAndPlaceState.GRASP,
                           f'Positioned above {self.current_cube_color} cube')
        else:
            self.get_logger().warn('Pre-grasp failed, returning to OBSERVE')
            self._transition(PickAndPlaceState.OBSERVE, 'Pre-grasp motion failed')

    def _state_grasp(self):
        """GRASP: Descend to cube and close gripper."""
        grasp_z_offset = self.get_parameter('offsets.grasp_z').value
        grasp_pos = self.current_cube_pos.copy()
        grasp_pos[2] += grasp_z_offset

        self.get_logger().info(f'Descending to GRASP at {grasp_pos} (world frame)')
        slow_duration = self.get_parameter('motion.slow_duration').value
        success = self._move_to_cartesian(grasp_pos, duration=slow_duration)

        if not success:
            self.get_logger().warn('Grasp descent failed')
            self._transition(PickAndPlaceState.OBSERVE, 'Grasp descent failed')
            return

        # Close gripper
        self._close_gripper()
        time.sleep(1.0)

        self._transition(PickAndPlaceState.VERIFY_GRASP, 'Gripper closed, verifying')

    def _state_verify_grasp(self):
        """VERIFY_GRASP: Lift up and check if cube is held."""
        post_grasp_z = self.get_parameter('offsets.post_grasp_z').value
        lift_pos = self.current_cube_pos.copy()
        lift_pos[2] += post_grasp_z

        self.get_logger().info(f'Lifting to {lift_pos} (world frame)')
        self._move_to_cartesian(lift_pos, duration=2.0)
        time.sleep(1.5)

        # Verify grasp
        success, checks, reason = self.grasp_verifier.verify_grasp(
            self.current_finger_pos,
            self.latest_cube_detections,
            self.current_cube_color,
            self.current_cube_pos,
        )

        self.get_logger().info(f'Grasp verification: {success}, {checks}/2 checks, {reason}')

        if success:
            self._transition(PickAndPlaceState.TRANSIT_TO_PLACE,
                           f'Grasp verified ({checks} checks)')
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.get_logger().error(f'Max retries for {self.current_cube_color}')
                self.cubes_sorted.add(self.current_cube_color)
                self.retry_count = 0
                self._transition(PickAndPlaceState.OBSERVE, 'Max retries, skipping')
            else:
                self._open_gripper()
                self._transition(PickAndPlaceState.OBSERVE,
                               f'Grasp failed, retry {self.retry_count}/{self.max_retries}')

    def _state_transit_to_place(self):
        """TRANSIT_TO_PLACE: Move above container."""
        container_pos = self.current_container_pos.copy()

        # Transit height: safe Z above table
        transit_z = self.waypoints.get('transit_height_z', 1.05)
        transit_pos = np.array([container_pos[0], container_pos[1], transit_z])

        self.get_logger().info(f'Transit to container at {transit_pos} (world frame)')
        success = self._move_to_cartesian(transit_pos, duration=3.0)

        # Drop check
        if self.drop_detector.check_for_drop(self.current_finger_pos):
            self.get_logger().warn('DROP DETECTED during transit!')
            self._transition(PickAndPlaceState.RECOVER_DROP, 'Drop during transit')
            return

        if not success:
            self._transition(PickAndPlaceState.RECOVER_DROP, 'Transit motion failed')
            return

        # Pre-place: above container
        pre_place_z = container_pos[2] + self.get_parameter('offsets.pre_place_z').value
        pre_place_pos = np.array([container_pos[0], container_pos[1], pre_place_z])

        self._move_to_cartesian(pre_place_pos, duration=2.0)

        if self.drop_detector.check_for_drop(self.current_finger_pos):
            self.get_logger().warn('DROP above container!')
            self._transition(PickAndPlaceState.RECOVER_DROP, 'Drop above container')
            return

        self._transition(PickAndPlaceState.PLACE, 'Above container')

    def _state_place(self):
        """PLACE: Descend into container and release."""
        container_pos = self.current_container_pos.copy()
        place_z = container_pos[2] + self.get_parameter('offsets.place_z').value
        place_pos = np.array([container_pos[0], container_pos[1], place_z])

        self.get_logger().info(f'Placing at {place_pos} (world frame)')
        slow_duration = self.get_parameter('motion.slow_duration').value
        self._move_to_cartesian(place_pos, duration=slow_duration)

        # Release
        self._open_gripper()
        time.sleep(0.5)

        # Retreat upward
        retreat_z = container_pos[2] + self.get_parameter('offsets.pre_place_z').value
        retreat_pos = np.array([container_pos[0], container_pos[1], retreat_z])
        self._move_to_cartesian(retreat_pos, duration=2.0)

        self._transition(PickAndPlaceState.VERIFY_PLACE,
                        f'Released {self.current_cube_color} cube')

    def _state_verify_place(self):
        """VERIFY_PLACE: Mark cube as sorted."""
        time.sleep(2.0)

        self.cubes_sorted[self.current_cube_color] = self.cubes_sorted.get(self.current_cube_color, 0) + 1
        self.retry_count = 0
        total_sorted = sum(self.cubes_sorted.values())
        self.get_logger().info(
            f'{self.current_cube_color} cube sorted! '
            f'Total: {total_sorted}/{self.total_cubes} ({self.cubes_sorted})'
        )

        if total_sorted >= self.total_cubes:
            self._transition(PickAndPlaceState.COMPLETE, f'All {self.total_cubes} cubes sorted')
        else:
            remaining = self.total_cubes - total_sorted
            self._transition(PickAndPlaceState.OBSERVE,
                           f'{remaining} cubes remaining')

    def _state_recover_drop(self):
        """RECOVER_DROP: Return to home and re-observe."""
        self.get_logger().warn('Running drop recovery...')
        self._open_gripper()

        home_angles = self.waypoints.get('home', {}).get(
            'joint_angles', [0.0, -0.35, 1.4, 0.0, -1.05, 0.0]
        )
        self._move_to_joint_angles(home_angles, duration=3.0)
        time.sleep(2.0)

        self._transition(PickAndPlaceState.OBSERVE, 'Recovery done, re-observing')

    def _state_complete(self):
        """COMPLETE: Return to home and log success."""
        home_angles = self.waypoints.get('home', {}).get(
            'joint_angles', [0.0, -0.35, 1.4, 0.0, -1.05, 0.0]
        )
        self._move_to_joint_angles(home_angles, duration=3.0)

        self.get_logger().info('=' * 60)
        self.get_logger().info('PICK-AND-PLACE COMPLETE!')
        self.get_logger().info(f'Sorted: {self.cubes_sorted}')
        self.get_logger().info('=' * 60)

        self.state = 'IDLE'


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
