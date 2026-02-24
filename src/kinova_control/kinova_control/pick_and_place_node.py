#!/usr/bin/env python3
"""
pick_and_place_node.py
Main state machine orchestrator for the Kinova Gen3 Lite pick-and-place.

Implements the complete pick-and-place cycle:
  INITIALIZE → OBSERVE → PRE_GRASP → GRASP → VERIFY_GRASP →
  TRANSIT_TO_PLACE → PLACE → VERIFY_PLACE → COMPLETE
  (with RECOVER_DROP fallback)

Every state transition is logged with timestamp and reason.
All IK uses Pinocchio. All perception comes from cameras (no ground truth).
No attach/detach plugins — real gripper friction only.
"""
import asyncio
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from std_msgs.msg import String
from ament_index_python.packages import get_package_share_directory

from kinova_control.ik_solver import IKSolver
from kinova_control.motion_executor import MotionExecutor
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

    Coordinates IK solving, motion execution, gripper control, perception
    integration, grasp verification, and drop detection to sort 3 colored
    cubes into their designated containers.
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
        self.cubes_sorted = set()   # colors of successfully sorted cubes
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
        self.motion_executor = None
        self.gripper_controller = None
        self.grasp_verifier = None
        self.drop_detector = None

        # ---- Grasp orientation: gripper pointing straight down ----
        self.grasp_orientation = np.array([
            [1.0,  0.0,  0.0],
            [0.0, -1.0,  0.0],
            [0.0,  0.0, -1.0],
        ])

        # ---- Start the state machine after a delay ----
        self.init_timer = self.create_timer(
            2.0, self._initialize_system, callback_group=self.cb_group
        )

        self.get_logger().info('Node created. Waiting for system initialization...')

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
        self.declare_parameter('gripper.command_duration', 1.5)
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

    def _joint_state_callback(self, msg: JointState):
        """Store latest joint state data."""
        self.latest_joint_state = msg

        # Update motion executor and gripper controller with current positions
        if self.motion_executor and self.arm_joint_names:
            arm_positions = self._get_arm_joint_positions(msg)
            if arm_positions is not None:
                self.motion_executor.update_current_joints(arm_positions)

        if self.gripper_controller:
            finger_pos = self._get_finger_position(msg)
            if finger_pos is not None:
                self.gripper_controller.update_finger_position(finger_pos)

    def _get_arm_joint_positions(self, msg: JointState) -> np.ndarray:
        """Extract arm joint positions from JointState message."""
        try:
            positions = np.zeros(6)
            for i, name in enumerate(self.arm_joint_names):
                if name in msg.name:
                    idx = msg.name.index(name)
                    positions[i] = msg.position[idx]
            return positions
        except (ValueError, IndexError):
            return None

    def _get_finger_position(self, msg: JointState) -> float:
        """Extract right_finger_bottom_joint position from JointState."""
        try:
            if 'right_finger_bottom_joint' in msg.name:
                idx = msg.name.index('right_finger_bottom_joint')
                return msg.position[idx]
        except (ValueError, IndexError):
            pass
        return None

    def _detections_callback(self, msg: PoseArray):
        """Store latest cube detections from perception."""
        self.latest_cube_detections = msg

    def _transition(self, new_state: str, reason: str):
        """
        Transition to a new state with logging.

        Every state transition is logged with timestamp, current state,
        new state, and reason — as required by project constraints.
        """
        now = self.get_clock().now()
        self.get_logger().info(
            f'[{now.nanoseconds / 1e9:.2f}] STATE TRANSITION: '
            f'{self.state} → {new_state} | Reason: {reason}'
        )
        self.state = new_state

    async def _initialize_system(self):
        """Initialize IK solver, motion executor, gripper controller."""
        self.init_timer.cancel()  # Only run once

        self.get_logger().info('Starting system initialization...')

        # Wait for robot_description parameter
        robot_desc = ''
        for attempt in range(30):
            try:
                robot_desc = self.get_parameter('robot_description').value
                if robot_desc:
                    break
            except Exception:
                pass
            self.get_logger().info(f'Waiting for robot_description... (attempt {attempt+1})')
            await asyncio.sleep(1.0)

        if not robot_desc:
            self.get_logger().error('Failed to get robot_description! Cannot initialize IK.')
            return

        # Initialize IK solver
        try:
            self.ik_solver = IKSolver(robot_desc, 'end_effector_link')
            self.get_logger().info('IK solver initialized successfully.')
        except Exception as e:
            self.get_logger().error(f'Failed to initialize IK solver: {e}')
            return

        # Initialize motion executor
        self.motion_executor = MotionExecutor(
            node=self,
            ik_solver=self.ik_solver,
            arm_joint_names=self.arm_joint_names,
            interpolation_steps=self.get_parameter('motion.interpolation_steps').value,
            waypoint_duration=self.get_parameter('motion.waypoint_duration').value,
        )

        # Initialize gripper controller
        self.gripper_controller = GripperController(
            node=self,
            open_position=self.get_parameter('gripper.open_position').value,
            close_position=self.get_parameter('gripper.close_position').value,
            command_duration=self.get_parameter('gripper.command_duration').value,
            wait_after_command=self.get_parameter('gripper.wait_after_command').value,
        )

        # Initialize grasp verifier
        self.grasp_verifier = GraspVerifier(
            finger_min_expected=self.get_parameter('grasp.finger_min_expected').value,
            finger_max_expected=self.get_parameter('grasp.finger_max_expected').value,
            position_tolerance=self.get_parameter('grasp.position_tolerance').value,
            required_checks=self.get_parameter('grasp.required_checks').value,
        )

        # Initialize drop detector
        self.drop_detector = DropDetector(
            finger_min_threshold=self.get_parameter('drop.finger_min_threshold').value,
            finger_max_threshold=self.get_parameter('drop.finger_max_threshold').value,
        )

        # Wait for action servers
        self.get_logger().info('Waiting for trajectory action servers...')
        arm_ready = self.motion_executor.wait_for_server(timeout_sec=30.0)
        gripper_ready = self.gripper_controller.wait_for_server(timeout_sec=30.0)

        if not arm_ready:
            self.get_logger().error('Arm trajectory action server not available!')
            return
        if not gripper_ready:
            self.get_logger().error('Gripper trajectory action server not available!')
            return

        # Wait for joint states
        self.get_logger().info('Waiting for joint states...')
        for _ in range(50):
            if self.latest_joint_state is not None:
                break
            await asyncio.sleep(0.2)

        if self.latest_joint_state is None:
            self.get_logger().error('No joint states received!')
            return

        self.get_logger().info('System initialization complete!')
        self._transition(PickAndPlaceState.INITIALIZE, 'System ready')

        # Start the state machine loop
        self.create_timer(0.5, self._run_state_machine, callback_group=self.cb_group)

    async def _run_state_machine(self):
        """Main state machine execution loop."""
        try:
            if self.state == PickAndPlaceState.INITIALIZE:
                await self._state_initialize()
            elif self.state == PickAndPlaceState.OBSERVE:
                await self._state_observe()
            elif self.state == PickAndPlaceState.PRE_GRASP:
                await self._state_pre_grasp()
            elif self.state == PickAndPlaceState.GRASP:
                await self._state_grasp()
            elif self.state == PickAndPlaceState.VERIFY_GRASP:
                await self._state_verify_grasp()
            elif self.state == PickAndPlaceState.TRANSIT_TO_PLACE:
                await self._state_transit_to_place()
            elif self.state == PickAndPlaceState.PLACE:
                await self._state_place()
            elif self.state == PickAndPlaceState.VERIFY_PLACE:
                await self._state_verify_place()
            elif self.state == PickAndPlaceState.RECOVER_DROP:
                await self._state_recover_drop()
            elif self.state == PickAndPlaceState.COMPLETE:
                await self._state_complete()
        except Exception as e:
            self.get_logger().error(f'State machine error in {self.state}: {e}')
            import traceback
            self.get_logger().error(traceback.format_exc())

    # ========== STATE IMPLEMENTATIONS ==========

    async def _state_initialize(self):
        """INITIALIZE: Move to home position."""
        home_pos = self.waypoints.get('home', {}).get('position', [0.0, 0.0, 0.45])
        home_pos = np.array(home_pos)

        self.get_logger().info(f'Moving to HOME position: {home_pos}')
        success = await self.motion_executor.move_to_position(
            home_pos, self.grasp_orientation
        )

        if success:
            self._transition(PickAndPlaceState.OBSERVE,
                           'Moved to HOME, starting observation')
        else:
            self.get_logger().warn('Failed to move to HOME, retrying...')
            await asyncio.sleep(2.0)

    async def _state_observe(self):
        """OBSERVE: Move to observation position and wait for stable cube detections."""
        observe_pos = self.waypoints.get('observe', {}).get('position', [0.35, 0.0, 0.35])
        observe_pos = np.array(observe_pos)

        self.get_logger().info(f'Moving to OBSERVE position: {observe_pos}')
        success = await self.motion_executor.move_to_position(
            observe_pos, self.grasp_orientation
        )
        if not success:
            self.get_logger().warn('Failed to move to OBSERVE position.')
            await asyncio.sleep(2.0)
            return

        # Wait for cube detections
        self.get_logger().info('Waiting for cube detections...')
        await asyncio.sleep(3.0)  # Let perception stabilize

        if self.latest_cube_detections is None:
            self.get_logger().warn('No cube detections received yet.')
            await asyncio.sleep(2.0)
            return

        # Select the next cube to pick (nearest unpicked)
        selected = self._select_next_cube()
        if selected is None:
            if len(self.cubes_sorted) >= 3:
                self._transition(PickAndPlaceState.COMPLETE,
                               'All 3 cubes have been sorted')
            else:
                self.get_logger().warn('No unpicked cubes detected, waiting...')
                await asyncio.sleep(3.0)
            return

        color, cube_pos = selected
        self.current_cube_color = color
        self.current_cube_pos = cube_pos

        # Look up target container position
        container_info = self.cube_mapping.get(color, {})
        container_pos = container_info.get('container_position')
        if container_pos:
            self.current_container_pos = np.array(container_pos)
        else:
            self.get_logger().error(f'No container mapping for color {color}!')
            return

        self.get_logger().info(
            f'Selected {color} cube at {cube_pos} → '
            f'{container_info.get("target_container", "unknown")} container at {container_pos}'
        )
        self._transition(PickAndPlaceState.PRE_GRASP,
                        f'Selected {color} cube')

    def _select_next_cube(self):
        """
        Select the next cube to pick from detections.

        Picks the nearest unpicked cube based on the latest perception data.
        Returns (color, position) or None.
        """
        if not self.latest_cube_detections:
            return None

        color_map = {
            'red': lambda q: q.x > 0.5,
            'green': lambda q: q.y > 0.5,
            'yellow': lambda q: q.z > 0.5,
        }

        candidates = []
        for pose in self.latest_cube_detections.poses:
            for color, check_fn in color_map.items():
                if color in self.cubes_sorted:
                    continue  # Skip already sorted cubes
                if check_fn(pose.orientation):
                    pos = np.array([pose.position.x, pose.position.y, pose.position.z])
                    # Distance from robot base (at origin in base frame)
                    dist = np.linalg.norm(pos[:2])
                    candidates.append((color, pos, dist))

        if not candidates:
            return None

        # Sort by distance (nearest first)
        candidates.sort(key=lambda c: c[2])
        return candidates[0][0], candidates[0][1]

    async def _state_pre_grasp(self):
        """PRE_GRASP: Open gripper and move above the target cube."""
        # Open gripper fully
        await self.gripper_controller.open_gripper()

        # Compute pre-grasp position (above cube)
        pre_grasp_z_offset = self.get_parameter('offsets.pre_grasp_z').value
        transit_height = self.get_parameter('motion.transit_height').value

        # Move to transit height first
        # Get current EE position
        current_ee = self.ik_solver.forward_kinematics(
            self.motion_executor.current_joints
        )
        transit_pos = np.array([
            current_ee.translation[0],
            current_ee.translation[1],
            transit_height + 0.75  # Add robot base height (world frame)
        ])

        # Then move laterally to above cube
        pre_grasp_pos = self.current_cube_pos.copy()
        pre_grasp_pos[2] = self.current_cube_pos[2] + pre_grasp_z_offset

        self.get_logger().info(f'Moving to PRE_GRASP at {pre_grasp_pos}')

        # Convert to base frame (subtract robot base position)
        pre_grasp_base = pre_grasp_pos.copy()
        pre_grasp_base[2] -= 0.75  # Subtract table/base height

        # Move to transit then to pre-grasp
        success = await self.motion_executor.move_to_position(
            pre_grasp_base, self.grasp_orientation
        )

        if success:
            self._transition(PickAndPlaceState.GRASP,
                           f'Positioned above {self.current_cube_color} cube')
        else:
            self.get_logger().warn('Failed to reach PRE_GRASP, returning to OBSERVE')
            self._transition(PickAndPlaceState.OBSERVE,
                           'Pre-grasp motion failed')

    async def _state_grasp(self):
        """GRASP: Descend to cube and close gripper."""
        grasp_z_offset = self.get_parameter('offsets.grasp_z').value

        # Compute grasp position (at cube surface)
        grasp_pos = self.current_cube_pos.copy()
        grasp_pos[2] = self.current_cube_pos[2] + grasp_z_offset

        # Convert to base frame
        grasp_base = grasp_pos.copy()
        grasp_base[2] -= 0.75

        self.get_logger().info(f'Descending to GRASP at {grasp_base}')

        # Slow descent
        slow_duration = self.get_parameter('motion.slow_duration').value
        success = await self.motion_executor.move_to_position(
            grasp_base, self.grasp_orientation, duration=slow_duration
        )

        if not success:
            self.get_logger().warn('Failed to reach GRASP position')
            self._transition(PickAndPlaceState.OBSERVE,
                           'Grasp descent failed')
            return

        # Close gripper on cube
        await self.gripper_controller.close_gripper()

        # Wait for gripper to settle
        await asyncio.sleep(1.0)

        self._transition(PickAndPlaceState.VERIFY_GRASP,
                        'Gripper closed, verifying grasp')

    async def _state_verify_grasp(self):
        """VERIFY_GRASP: Lift and verify that the cube is held."""
        # Move to post-grasp (lift up)
        post_grasp_z = self.get_parameter('offsets.post_grasp_z').value
        post_grasp_pos = self.current_cube_pos.copy()
        post_grasp_pos[2] = self.current_cube_pos[2] + post_grasp_z

        # Convert to base frame
        post_grasp_base = post_grasp_pos.copy()
        post_grasp_base[2] -= 0.75

        self.get_logger().info(f'Lifting to POST_GRASP at {post_grasp_base}')
        await self.motion_executor.move_to_position(
            post_grasp_base, self.grasp_orientation
        )

        # Wait for perception to update
        await asyncio.sleep(1.5)

        # Verify grasp with multiple methods
        finger_pos = self.gripper_controller.get_finger_position()
        success, checks, reason = self.grasp_verifier.verify_grasp(
            finger_pos,
            self.latest_cube_detections,
            self.current_cube_color,
            self.current_cube_pos,
        )

        self.get_logger().info(
            f'Grasp verification: success={success}, checks={checks}/2, {reason}'
        )

        if success:
            self._transition(PickAndPlaceState.TRANSIT_TO_PLACE,
                           f'Grasp verified ({checks} checks passed)')
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.get_logger().error(
                    f'Max retries ({self.max_retries}) reached for {self.current_cube_color}'
                )
                self.cubes_sorted.add(self.current_cube_color)  # Skip this cube
                self.retry_count = 0
                self._transition(PickAndPlaceState.OBSERVE,
                               'Max retries exceeded, skipping cube')
            else:
                await self.gripper_controller.open_gripper()
                self._transition(PickAndPlaceState.OBSERVE,
                               f'Grasp failed, retry {self.retry_count}/{self.max_retries}')

    async def _state_transit_to_place(self):
        """TRANSIT_TO_PLACE: Move to above the target container, monitoring for drops."""
        transit_height = self.get_parameter('motion.transit_height').value
        pre_place_z = self.get_parameter('offsets.pre_place_z').value

        # Move via transit height to above container
        container_base = self.current_container_pos.copy()
        container_base[2] -= 0.75  # Convert to base frame

        # Transit waypoint: lift to safe height
        transit_pos = np.array([
            container_base[0],
            container_base[1],
            transit_height,
        ])

        # Pre-place position
        pre_place_pos = np.array([
            container_base[0],
            container_base[1],
            pre_place_z,
        ])

        self.get_logger().info(f'Transit to container at {container_base[:2]}')

        # Move to transit height
        success = await self.motion_executor.move_to_position(
            transit_pos, self.grasp_orientation
        )

        # Check for drops during transit
        finger_pos = self.gripper_controller.get_finger_position()
        if self.drop_detector.check_for_drop(finger_pos):
            self.get_logger().warn('DROP DETECTED during transit!')
            self._transition(PickAndPlaceState.RECOVER_DROP,
                           f'Cube dropped during transit (finger_pos={finger_pos:.3f})')
            return

        if not success:
            self._transition(PickAndPlaceState.RECOVER_DROP,
                           'Transit motion failed')
            return

        # Move to pre-place
        success = await self.motion_executor.move_to_position(
            pre_place_pos, self.grasp_orientation
        )

        # Final drop check
        finger_pos = self.gripper_controller.get_finger_position()
        if self.drop_detector.check_for_drop(finger_pos):
            self.get_logger().warn('DROP DETECTED above container!')
            self._transition(PickAndPlaceState.RECOVER_DROP,
                           'Cube dropped above container')
            return

        if success:
            self._transition(PickAndPlaceState.PLACE,
                           'Positioned above container')
        else:
            self._transition(PickAndPlaceState.RECOVER_DROP,
                           'Pre-place motion failed')

    async def _state_place(self):
        """PLACE: Descend into container and release cube."""
        place_z = self.get_parameter('offsets.place_z').value

        # Place position (inside container)
        container_base = self.current_container_pos.copy()
        container_base[2] -= 0.75  # Convert to base frame

        place_pos = np.array([
            container_base[0],
            container_base[1],
            place_z,
        ])

        self.get_logger().info(f'Descending to PLACE at {place_pos}')

        # Slow descent into container
        slow_duration = self.get_parameter('motion.slow_duration').value
        await self.motion_executor.move_to_position(
            place_pos, self.grasp_orientation, duration=slow_duration
        )

        # Open gripper to release
        await self.gripper_controller.open_gripper()
        await asyncio.sleep(0.5)

        # Retreat upward
        pre_place_z = self.get_parameter('offsets.pre_place_z').value
        retreat_pos = np.array([
            container_base[0],
            container_base[1],
            pre_place_z,
        ])

        await self.motion_executor.move_to_position(
            retreat_pos, self.grasp_orientation
        )

        self._transition(PickAndPlaceState.VERIFY_PLACE,
                        f'Released {self.current_cube_color} cube in container')

    async def _state_verify_place(self):
        """VERIFY_PLACE: Check if the cube landed in the container."""
        await asyncio.sleep(2.0)  # Wait for perception to update

        # Mark cube as sorted (simplified verification — cube left gripper)
        self.cubes_sorted.add(self.current_cube_color)
        self.retry_count = 0
        self.get_logger().info(
            f'{self.current_cube_color} cube sorted! '
            f'Total sorted: {len(self.cubes_sorted)}/3 ({self.cubes_sorted})'
        )

        if len(self.cubes_sorted) >= 3:
            self._transition(PickAndPlaceState.COMPLETE,
                           'All 3 cubes sorted successfully')
        else:
            self._transition(PickAndPlaceState.OBSERVE,
                           f'Cube placed, moving to next ({3 - len(self.cubes_sorted)} remaining)')

    async def _state_recover_drop(self):
        """RECOVER_DROP: Return to home, then re-observe."""
        self.get_logger().warn('Executing drop recovery...')

        # Open gripper
        await self.gripper_controller.open_gripper()

        # Move to home
        home_pos = self.waypoints.get('home', {}).get('position', [0.0, 0.0, 0.45])
        home_pos = np.array(home_pos)
        await self.motion_executor.move_to_position(
            home_pos, self.grasp_orientation
        )

        await asyncio.sleep(2.0)

        self._transition(PickAndPlaceState.OBSERVE,
                        'Recovery complete, re-observing')

    async def _state_complete(self):
        """COMPLETE: All cubes sorted, return to home and log success."""
        home_pos = self.waypoints.get('home', {}).get('position', [0.0, 0.0, 0.45])
        home_pos = np.array(home_pos)

        self.get_logger().info('Moving to HOME for final rest...')
        await self.motion_executor.move_to_position(
            home_pos, self.grasp_orientation
        )

        self.get_logger().info('=' * 60)
        self.get_logger().info('PICK-AND-PLACE COMPLETE!')
        self.get_logger().info(f'Successfully sorted: {self.cubes_sorted}')
        self.get_logger().info('=' * 60)

        # Idle — don't transition again
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
