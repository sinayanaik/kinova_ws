#!/usr/bin/env python3
"""
pick_and_place_node.py
State machine for Kinova Gen3 Lite pick-and-place.

The motion layer uses quintic time scaling for both joint-space and Cartesian
segments. Placement is verified from perception before a cube is counted as
sorted, so a cube that lands outside its tray is re-acquired instead of being
silently accepted.
"""
import time
import threading
import numpy as np
import yaml
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseArray
from ament_index_python.packages import get_package_share_directory

from kinova_control.ik_solver import IKSolver
from kinova_control.gripper_controller import GripperController
from kinova_control.grasp_verifier import GraspVerifier, DropDetector
from kinova_control.motion_executor import MotionExecutor


class S:
    """State names."""
    INITIALIZE      = 'INITIALIZE'
    OBSERVE         = 'OBSERVE'
    PRE_GRASP       = 'PRE_GRASP'
    GRASP           = 'GRASP'
    VERIFY_GRASP    = 'VERIFY_GRASP'
    TRANSIT         = 'TRANSIT_TO_PLACE'
    PLACE           = 'PLACE'
    VERIFY_PLACE    = 'VERIFY_PLACE'
    RECOVER_DROP    = 'RECOVER_DROP'
    COMPLETE        = 'COMPLETE'


class PickAndPlaceNode(Node):

    def __init__(self):
        super().__init__('pick_and_place_node')
        self.get_logger().info('Pick-and-Place node starting...')
        self.cb_group = ReentrantCallbackGroup()
        self.run_started_at = time.monotonic()
        self.low_latency_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # ---------- ROS parameters ----------
        self._declare_params()

        # ---------- State machine ----------
        self.state = S.INITIALIZE
        self.cubes_sorted = {}          # color -> int
        self.total_cubes = 6
        self.current_cube_color = None
        self.current_cube_pos = None
        self.current_container_pos = None
        self.retry_count = 0
        self.max_retries = 3
        self.desired_wrist_angle = None
        self.no_target_count = 0
        self.max_no_target_cycles = 3
        self.last_state = None
        self.last_detection_time = 0.0

        # Ordered picking: red -> green -> yellow -> red -> green -> yellow
        self.pick_order = ['red', 'green', 'yellow', 'red', 'green', 'yellow']
        self.pick_index = 0

        # ---------- Sensor data ----------
        self.latest_joint_state = None
        self.latest_cube_detections = None
        self.arm_joint_names = None
        self.current_arm_positions = np.zeros(6)
        self.commanded_arm_positions = np.zeros(6)
        self.current_finger_pos = 0.0

        # Load configs
        self._load_configs()

        # Targets per color from config (fallback to 2 each)
        self.cube_target_counts = {
            c: self.cube_mapping.get(c, {}).get('num_cubes', 2)
            for c in ['red', 'green', 'yellow']
        }
        self.total_cubes = sum(self.cube_target_counts.values())

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_cb, 10, callback_group=self.cb_group)
        self.create_subscription(
            PoseArray, '/perception/cube_detections',
            self._det_cb, self.low_latency_qos, callback_group=self.cb_group)

        # URDF for Pinocchio
        self.declare_parameter('robot_description', '')
        self.ik_solver = None
        self.motion_executor = MotionExecutor(
            node=self,
            ik_solver=None,
            arm_joint_names=self.arm_joint_names,
            sample_period=self._param_float('motion.sample_period'),
            min_samples=self._param_int('motion.minimum_samples'),
            max_cartesian_step=self._param_float('motion.cartesian_max_step'),
            joint_velocity_limit=self._param_float('motion.joint_velocity_limit'),
            joint_acceleration_limit=self._param_float('motion.joint_acceleration_limit'),
        )
        self.gripper_controller = GripperController(
            node=self,
            open_position=self.get_parameter('gripper.open_position').value,
            close_position=self.get_parameter('gripper.close_position').value,
            max_effort=self.get_parameter('gripper.max_effort').value,
            wait_after_command=self.get_parameter('gripper.wait_after_command').value)
        self.grasp_verifier = None
        self.drop_detector = None

        # Grasp orientation: tool_frame pointing straight down
        # tool_frame has Rz(π/2) offset from end_effector_link
        self.grasp_R = np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0]])

        # Start state-machine thread
        self._stop_event = threading.Event()
        self._sm_thread = threading.Thread(target=self._sm_loop, daemon=True)
        self._sm_thread.start()

    # ============================================================
    # PARAMETERS
    # ============================================================
    def _declare_params(self):
        p = self.declare_parameter
        p('ik.max_iterations', 300)
        p('ik.tolerance', 0.002)
        p('ik.damping_lambda', 1e-4)
        p('ik.step_size', 0.5)
        p('motion.sample_period', 0.015)
        p('motion.minimum_samples', 16)
        p('motion.cartesian_max_step', 0.006)
        p('motion.cartesian_speed', 0.22)
        p('motion.cartesian_min_duration', 0.12)
        p('motion.joint_velocity_limit', 2.5)
        p('motion.joint_acceleration_limit', 6.0)
        p('motion.waypoint_duration', 0.35)
        p('motion.slow_duration', 0.50)
        p('motion.home_duration', 0.45)
        p('motion.observe_duration', 0.20)
        p('motion.pre_grasp_duration', 0.28)
        p('motion.coarse_grasp_duration', 0.18)
        p('motion.grasp_cartesian_duration', 0.40)
        p('motion.post_grasp_lift_duration', 0.25)
        p('motion.transit_duration', 0.35)
        p('motion.place_descent_duration', 0.25)
        p('motion.place_retreat_duration', 0.18)
        p('motion.recover_home_duration', 0.70)
        p('motion.complete_home_duration', 0.90)
        p('motion.transit_height', 0.35)
        p('motion.transit_clearance_z', 0.92)
        p('motion.transit_stage_x_mid', 0.30)
        p('motion.transit_stage_x_target', 0.35)
        p('motion.transit_stage_duration', 0.16)
        p('timing.observe_settle', 0.04)
        p('timing.observe_skip_pause', 0.02)
        p('timing.no_detection_wait', 0.20)
        p('timing.no_target_wait', 0.20)
        p('timing.verify_observe_settle', 0.08)
        p('timing.pre_grasp_retry_pause', 0.05)
        p('timing.coarse_grasp_settle', 0.04)
        p('timing.pre_close_settle', 0.08)
        p('timing.post_close_hold', 0.25)
        p('timing.post_lift_settle', 0.02)
        p('timing.pre_release_settle', 0.04)
        p('timing.post_release_settle', 0.08)
        p('timing.post_place_settle', 0.02)
        p('timing.recover_settle', 0.10)
        p('placement.finger_angle', 0.0)  # fingers aligned with tray long axis
        p('placement.slot_axis', 'x')
        p('placement.verify_timeout', 1.60)
        p('placement.verify_poll_period', 0.04)
        p('gripper.open_position', 0.0)
        p('gripper.close_position', 0.82)
        p('gripper.max_effort', 100.0)
        p('gripper.wait_after_command', 0.08)
        p('gripper.command_timeout', 1.0)
        p('gripper.poll_period', 0.02)
        p('gripper.position_tolerance', 0.02)
        p('gripper.stable_epsilon', 0.002)
        p('gripper.stable_cycles', 4)
        p('grasp.finger_min_expected', 0.15)
        p('grasp.finger_max_expected', 0.75)
        p('grasp.position_tolerance', 0.05)
        p('grasp.required_checks', 1)
        p('grasp.max_z_axis_error', 0.06)
        p('grasp.max_z_axis_error_coarse', 0.10)
        p('drop.finger_min_threshold', 0.1)
        p('drop.finger_max_threshold', 0.84)
        p('selection.allow_color_fallback', False)
        p('selection.detection_timeout', 0.60)
        p('selection.observe_joint_tolerance', 0.05)
        p('selection.max_retarget_distance', 0.06)
        p('selection.source_clear_x_threshold', 0.30)
        p('workspace.x_min', 0.10)
        p('workspace.x_max', 0.48)
        p('workspace.y_limit', 0.28)
        p('offsets.pre_grasp_z', 0.05)
        p('offsets.grasp_z', -0.006)
        p('offsets.post_grasp_z', 0.08)
        p('offsets.pre_place_z', 0.10)
        p('offsets.drop_height', 0.07)  # height above container to open gripper and drop
        p('offsets.place_release_z', 0.021)  # tray floor + cube_half_height + clearance
        p('tray.inner_size_x', 0.155)
        p('tray.inner_size_y', 0.115)
        p('tray.wall_clearance', 0.016)
        p('tray.verify_margin', 0.012)
        p('cube.size', 0.035)
        p('arm_joint_names',
          ['joint_1','joint_2','joint_3','joint_4','joint_5','joint_6'])
        p('gripper_joint_names',
          ['right_finger_bottom_joint','right_finger_tip_joint',
           'left_finger_bottom_joint','left_finger_tip_joint'])

    def _load_configs(self):
        desc = get_package_share_directory('kinova_description')
        try:
            with open(f'{desc}/config/cube_container_mapping.yaml') as f:
                d = yaml.safe_load(f)
            self.cube_mapping = d.get('cube_container_mapping', {})
        except Exception as e:
            self.get_logger().error(f'cube_container_mapping: {e}')
            self.cube_mapping = {}
        try:
            with open(f'{desc}/config/waypoints.yaml') as f:
                d = yaml.safe_load(f)
            self.waypoints = d.get('waypoints', {})
        except Exception as e:
            self.get_logger().error(f'waypoints: {e}')
            self.waypoints = {}
        self.arm_joint_names = self.get_parameter('arm_joint_names').value

    # ============================================================
    # CALLBACKS
    # ============================================================
    def _joint_state_cb(self, msg):
        self.latest_joint_state = msg
        try:
            for i, name in enumerate(self.arm_joint_names):
                if name in msg.name:
                    self.current_arm_positions[i] = msg.position[msg.name.index(name)]
            if 'right_finger_bottom_joint' in msg.name:
                self.current_finger_pos = msg.position[
                    msg.name.index('right_finger_bottom_joint')]
        except (ValueError, IndexError):
            pass

    def _det_cb(self, msg):
        self.latest_cube_detections = msg
        self.last_detection_time = time.time()

    def _elapsed_tag(self):
        return f'[t={time.monotonic() - self.run_started_at:6.2f}s]'

    def _log(self, txt):
        self.get_logger().info(f'{self._elapsed_tag()} {txt}')

    def _transition(self, new, reason):
        self.last_state = self.state
        self._log(f'STATE {self.state} -> {new} | {reason}')
        self.state = new

    def destroy_node(self):
        self._stop_event.set()
        sm_thread = getattr(self, '_sm_thread', None)
        if sm_thread is not None and sm_thread.is_alive() and threading.current_thread() is not sm_thread:
            sm_thread.join(timeout=6.0)
        return super().destroy_node()

    def _should_stop(self):
        return self._stop_event.is_set() or not rclpy.ok()

    def _sleep(self, duration):
        deadline = time.monotonic() + max(0.0, float(duration))
        while not self._should_stop():
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return True
            time.sleep(min(0.05, remaining))
        return False

    def _param_float(self, name):
        return float(self.get_parameter(name).value)

    def _param_int(self, name):
        return int(self.get_parameter(name).value)

    def _param_bool(self, name):
        return bool(self.get_parameter(name).value)

    def _observe_joints(self):
        return np.array(
            self.waypoints.get('observe', {}).get(
                'joint_angles', [0.62, -0.82, 0.75, 1.57, 1.57, 1.64]
            ),
            dtype=float,
        )

    def _home_joints(self):
        return np.array(
            self.waypoints.get('home', {}).get('joint_angles', [0.0] * 6),
            dtype=float,
        )

    def _is_near_joint_target(self, target, tolerance=None):
        target = np.array(target, dtype=float)
        if tolerance is None:
            tolerance = self._param_float('selection.observe_joint_tolerance')
        return float(np.max(np.abs(self.current_arm_positions - target))) <= float(tolerance)

    def _tool_translation(self):
        if self.ik_solver is None:
            return None
        try:
            return self.ik_solver.forward_kinematics(self.commanded_arm_positions).translation.copy()
        except Exception:
            return None

    def _detections_are_fresh(self, timeout=None):
        if self.latest_cube_detections is None or self.last_detection_time <= 0.0:
            return False
        if timeout is None:
            timeout = self._param_float('selection.detection_timeout')
        return (time.time() - self.last_detection_time) <= float(timeout)

    def _clear_active_target(self, reset_color=False):
        if reset_color:
            self.current_cube_color = None
        self.current_cube_pos = None
        self.current_container_pos = None
        self.desired_wrist_angle = None

    def _is_source_row_clear(self):
        tool = self._tool_translation()
        if tool is None:
            return False
        return float(tool[0]) >= self._param_float('selection.source_clear_x_threshold')

    def _verify_clear_joints(self):
        q = self._observe_joints().copy()
        q[0] = self.current_arm_positions[0]
        q[5] = self.current_arm_positions[5]
        return q

    def _sync_commanded_arm_state(self, joint_positions):
        q = np.array(joint_positions, dtype=float).copy()
        self.commanded_arm_positions = q
        if self.motion_executor is not None:
            self.motion_executor.update_current_joints(q)

    def _move_joints(self, target, dur=None):
        """Move with an analytical quintic joint-space trajectory."""
        target = np.array(target, dtype=float)
        if self._is_near_joint_target(target, tolerance=1e-3):
            return True
        if dur is None:
            dur = self._param_float('motion.waypoint_duration')
        if self.motion_executor is None:
            self.get_logger().error('Motion executor not ready')
            return False
        traj = self.motion_executor.plan_joint_trajectory(
            target,
            float(dur),
            q_init=self.commanded_arm_positions.copy(),
        )
        ok = self.motion_executor.execute_trajectory_blocking(traj)
        if ok:
            self._sync_commanded_arm_state(target)
        return ok

    def _move_through_poses(self, pose_list, dur_list, strict=False):
        """Smooth joint-space move through multiple Cartesian poses.

        Solves IK for each pose, then uses cubic Hermite multi-waypoint
        planner so the robot passes through intermediates at non-zero
        velocity (no halting).
        """
        if self.ik_solver is None or self.motion_executor is None:
            return False

        joint_waypoints = []
        q_prev = self.commanded_arm_positions.copy()
        for pos in pose_list:
            pos = np.array(pos, dtype=float)
            ok, q = self.ik_solver.solve_robust(
                pos, self.grasp_R, q_prev.copy(), 300, 2e-3)
            ee = self.ik_solver.forward_kinematics(q)
            err = np.linalg.norm(ee.translation - pos)
            threshold = 0.008 if strict else 0.015
            if err > threshold:
                ok2, q2 = self.ik_solver.solve_z_axis_down(
                    pos, q_prev.copy(), 500, 2e-3)
                ee2 = self.ik_solver.forward_kinematics(q2)
                err2 = np.linalg.norm(ee2.translation - pos)
                if err2 < err:
                    q = q2
            # Apply wrist angle if set
            if self.desired_wrist_angle is not None:
                q = self._align_fingers_to_angle(q, self.desired_wrist_angle)
            elif self.current_cube_pos is not None:
                q = self._align_fingers_avoid_neighbors(q, self.current_cube_pos)
            q = self.ik_solver.clamp_arm_joints(q)
            joint_waypoints.append(q)
            q_prev = q

        traj = self.motion_executor.plan_multi_waypoint_trajectory(
            joint_waypoints, dur_list,
            q_init=self.commanded_arm_positions.copy(),
        )
        ok = self.motion_executor.execute_trajectory_blocking(traj)
        if ok and joint_waypoints:
            self._sync_commanded_arm_state(joint_waypoints[-1])
        return ok

    def _cartesian_segment_durations(self, points, minimum_duration=None):
        if not points:
            return []

        cartesian_speed = max(self._param_float('motion.cartesian_speed'), 1e-3)
        min_duration = self._param_float('motion.cartesian_min_duration')
        if minimum_duration is not None:
            min_duration = max(min_duration, float(minimum_duration))

        tool = self._tool_translation()
        if tool is None:
            tool = np.array(points[0], dtype=float)

        durations = []
        prev = np.array(tool, dtype=float)
        for waypoint in points:
            waypoint = np.array(waypoint, dtype=float)
            distance = float(np.linalg.norm(waypoint - prev))
            durations.append(max(min_duration, distance / cartesian_speed))
            prev = waypoint
        return durations

    def _append_waypoint(self, points, pos, tol=0.01):
        waypoint = np.array(pos, dtype=float)
        if points and np.linalg.norm(points[-1] - waypoint) <= tol:
            return
        points.append(waypoint)

    def _move_cartesian_sequence(self, waypoints, segment_duration, strict_fallback=False, label='Path'):
        points = []
        for waypoint in waypoints:
            self._append_waypoint(points, waypoint)
        if not points:
            return True

        segment_durations = self._cartesian_segment_durations(
            points,
            minimum_duration=segment_duration,
        )
        total = len(points)
        for idx, (waypoint, duration) in enumerate(zip(points, segment_durations), start=1):
            self._log(
                f'{label} stage {idx}/{total}: {np.round(waypoint, 3)} '
                f'({duration:.2f}s)'
            )

        def postprocess_q(q):
            if self.desired_wrist_angle is not None:
                return self._align_fingers_to_angle(q, self.desired_wrist_angle)
            return self._align_fingers_tangential(q)

        traj = self.motion_executor.plan_cartesian_polyline(
            np.array(points, dtype=float),
            segment_duration=segment_durations,
            target_orientation=self.grasp_R,
            q_init=self.commanded_arm_positions.copy(),
            max_cartesian_step=self._param_float('motion.cartesian_max_step'),
            ik_tolerance=self._param_float('ik.tolerance'),
            postprocess_q=postprocess_q,
        )
        if traj is not None:
            ok = self.motion_executor.execute_trajectory_blocking(traj)
            if ok:
                self._sync_commanded_arm_state(np.array(traj.points[-1].positions, dtype=float))
                return True
            self._log(f'{label} polyline execution failed, falling back to segmented execution')
        else:
            self._log(f'{label} polyline planning failed, falling back to segmented execution')

        for idx, (waypoint, duration) in enumerate(zip(points, segment_durations), start=1):
            ok = self._move_cartesian(waypoint, dur=duration)
            if not ok:
                ok = self._move_to_pose(
                    waypoint,
                    dur=duration,
                    strict=not strict_fallback,
                )
            if not ok:
                self._log(f'{label} stage {idx}/{total} failed')
                return False
        return True

    def _build_transit_waypoints(self, place_target):
        tool = self._tool_translation()
        if tool is None:
            return []

        pre_place_z = self._param_float('offsets.pre_place_z')
        z_clear = max(
            self._param_float('motion.transit_clearance_z'),
            float(tool[2]),
            float(place_target[2] + pre_place_z),
        )
        x_mid = self._param_float('motion.transit_stage_x_mid')
        x_target = self._param_float('motion.transit_stage_x_target')
        above = np.array(place_target, dtype=float).copy()
        above[2] += pre_place_z

        waypoints = []
        self._append_waypoint(waypoints, [tool[0], tool[1], z_clear], tol=0.005)
        self._append_waypoint(waypoints, [x_mid, tool[1], z_clear])
        self._append_waypoint(waypoints, [x_mid, above[1], z_clear])
        self._append_waypoint(waypoints, [x_target, above[1], z_clear])
        self._append_waypoint(waypoints, above)
        return waypoints

    def _move_to_pose(self, pos, dur=None, strict=True):
        """IK + joint-space move with bounded position and Z-axis error.
        
        Args:
            pos: Target position [x, y, z]
            dur: Move duration
            strict: If True, require precise position match (err < 0.008m).
                   If False, accept looser match (err < 0.015m) for coarse approach.
        """
        if self.ik_solver is None:
            return False
        if dur is None:
            dur = self._param_float('motion.waypoint_duration')
        ok, q = self.ik_solver.solve_robust(
            np.array(pos), self.grasp_R,
            self.commanded_arm_positions.copy(), 300, 2e-3)
        ee = self.ik_solver.forward_kinematics(q)
        err = np.linalg.norm(ee.translation - np.array(pos))

        # Check orientation: tool Z should point down
        tool_z = ee.rotation[:, 2]
        z_err = np.linalg.norm(tool_z - np.array([0, 0, -1]))

        err_threshold = 0.008 if strict else 0.015
        z_threshold = (
            self._param_float('grasp.max_z_axis_error')
            if strict else self._param_float('grasp.max_z_axis_error_coarse')
        )
        if err > err_threshold or z_err > z_threshold:
            # Try Z-axis-down fallback (5D: position + vertical orientation)
            self._log(
                f'IK err={err:.4f}, z_err={z_err:.4f}, trying Z-down 5D (strict={strict})...'
            )
            ok2, q2 = self.ik_solver.solve_z_axis_down(
                np.array(pos),
                self.commanded_arm_positions.copy(), 500, 2e-3)
            ee2 = self.ik_solver.forward_kinematics(q2)
            err2 = np.linalg.norm(ee2.translation - np.array(pos))
            z_err2 = np.linalg.norm(ee2.rotation[:, 2] - np.array([0, 0, -1]))
            if err2 < err:
                q, ee, err, z_err = q2, ee2, err2, z_err2
            
            # Final check
            fail_threshold = 0.012 if strict else 0.020
            if err > fail_threshold or z_err > z_threshold:
                if strict:
                    self.get_logger().warn(
                        f'IK FAIL target={np.round(pos,3)} '
                        f'achieved={np.round(ee.translation,3)} err={err:.4f} z_err={z_err:.4f}')
                    return False
                else:
                    self._log(f'IK loose match rejected: err={err:.4f} z_err={z_err:.4f}')
                    return False
        
        # Use an explicit wrist angle when one has already been chosen.
        if self.desired_wrist_angle is not None:
            q = self._align_fingers_to_angle(q, self.desired_wrist_angle)
        elif self.current_cube_pos is not None:
            q = self._align_fingers_avoid_neighbors(q, self.current_cube_pos)
        else:
            q = self._align_fingers_tangential(q)
        q = self.ik_solver.clamp_arm_joints(q)
        ee = self.ik_solver.forward_kinematics(q)
        err = np.linalg.norm(ee.translation - np.array(pos))
        z_err = np.linalg.norm(ee.rotation[:, 2] - np.array([0, 0, -1]))
        self._log(f'IK OK target={np.round(pos,3)} '
                  f'achieved={np.round(ee.translation,3)} err={err:.4f} z_err={z_err:.4f}')
        return self._move_joints(q, dur)

    def _align_fingers_tangential(self, q):
        """Adjust j6 (wrist roll) so gripper fingers open along Y-axis.

        Since j6 only rotates around tool Z (≈ world -Z), changing it
        preserves position and Z-down orientation.  The relationship is:
        Δφ = -Δj6  (finger angle decreases as j6 increases).
        """
        se3 = self.ik_solver.forward_kinematics(q)
        finger = se3.rotation[:, 0]  # tool X = finger opening direction
        phi = np.arctan2(finger[1], finger[0])  # current angle in XY plane
        # Target: fingers along ±Y → φ = ±π/2
        # Pick closest target (±π/2)
        delta_plus  = ((np.pi / 2 - phi) + np.pi) % (2 * np.pi) - np.pi
        delta_minus = ((-np.pi / 2 - phi) + np.pi) % (2 * np.pi) - np.pi
        delta = delta_plus if abs(delta_plus) < abs(delta_minus) else delta_minus
        q_out = q.copy()
        q_out[5] -= delta  # Δφ = -Δj6, so j6 -= delta to achieve Δφ = +delta
        return q_out

    def _align_fingers_to_angle(self, q, target_angle):
        """Rotate the wrist so the finger opening direction matches target_angle."""
        se3 = self.ik_solver.forward_kinematics(q)
        finger = se3.rotation[:, 0]
        current_phi = np.arctan2(finger[1], finger[0])
        delta1 = ((target_angle - current_phi) + np.pi) % (2 * np.pi) - np.pi
        delta2 = ((target_angle + np.pi - current_phi) + np.pi) % (2 * np.pi) - np.pi
        delta = delta1 if abs(delta1) < abs(delta2) else delta2
        q_out = q.copy()
        q_out[5] -= delta
        return q_out

    def _angle_diff(self, a, b):
        return ((a - b) + np.pi) % (2 * np.pi) - np.pi

    def _current_finger_angle(self):
        if self.ik_solver is None:
            return None
        se3 = self.ik_solver.forward_kinematics(self.commanded_arm_positions)
        finger = se3.rotation[:, 0]
        return np.arctan2(finger[1], finger[0])

    def _get_all_detected_cube_positions(self):
        """Extract all currently detected cube positions from latest detections."""
        positions = []
        if not self._detections_are_fresh():
            return positions
        for pose in self.latest_cube_detections.poses:
            p = np.array([pose.position.x, pose.position.y, pose.position.z])
            positions.append(p)
        return positions

    def _compute_desired_wrist_angle(self, target_pos):
        """Compute wrist angle so fingers are PERPENDICULAR to the neighbor line.

        Algorithm:
          1. Find the closest neighbor on each side along the row.
          2. Draw a straight line through those two neighbors (or from
             the target to the single neighbor).
          3. Orient the finger opening direction PERPENDICULAR to that line
             so the open fingers fit in the gaps instead of sweeping into
             adjacent cubes.
          4. No neighbors → default fingers along X (perpendicular to Y row).
        """
        target_xy = target_pos[:2]
        all_cubes = self._get_all_detected_cube_positions()

        # Collect nearby cubes (within 12cm, excluding self)
        neighbors = []
        for cp in all_cubes:
            d = np.linalg.norm(cp[:2] - target_xy)
            if 0.02 < d < 0.12:
                neighbors.append(cp[:2])

        if not neighbors:
            # No neighbors — fingers along X (perpendicular to typical Y-row)
            self._log('  No neighbors — fingers along X (0°)')
            return 0.0

        # Sort neighbors by signed offset along Y relative to target
        # (left = negative Y, right = positive Y)
        by_y = sorted(neighbors, key=lambda nb: nb[1] - target_xy[1])

        left = None   # closest neighbor with smaller Y
        right = None  # closest neighbor with larger Y
        for nb in by_y:
            dy = nb[1] - target_xy[1]
            if dy < -0.01 and (left is None or abs(dy) < abs(left[1] - target_xy[1])):
                left = nb
            elif dy > 0.01 and (right is None or abs(dy) < abs(right[1] - target_xy[1])):
                right = nb

        # Compute the line direction
        if left is not None and right is not None:
            line_dir = right - left
            self._log(f'  Row line: left {np.round(left,3)} — right {np.round(right,3)}')
        elif left is not None:
            line_dir = target_xy - left
            self._log(f'  Single neighbor left: {np.round(left,3)}')
        elif right is not None:
            line_dir = right - target_xy
            self._log(f'  Single neighbor right: {np.round(right,3)}')
        else:
            # All neighbors are nearly co-located with target along Y
            self._log('  Neighbors co-located, fingers along X (0°)')
            return 0.0

        line_norm = np.linalg.norm(line_dir)
        if line_norm < 1e-6:
            self._log('  Degenerate line, fingers along X (0°)')
            return 0.0

        line_dir /= line_norm
        line_angle = np.arctan2(line_dir[1], line_dir[0])

        # Fingers must open PERPENDICULAR to the line (±90°)
        perp1 = line_angle + np.pi / 2
        perp2 = line_angle - np.pi / 2

        # Pick whichever is closer to current wrist angle (less rotation)
        prefer = self._current_finger_angle()
        if prefer is None:
            prefer = 0.0
        d1 = abs(self._angle_diff(perp1, prefer))
        d2 = abs(self._angle_diff(perp2, prefer))
        best = perp1 if d1 <= d2 else perp2
        # Wrap to [-π, π]
        best = ((best + np.pi) % (2 * np.pi)) - np.pi

        self._log(f'  Line angle: {np.degrees(line_angle):.0f}° → '
                  f'fingers perpendicular: {np.degrees(best):.0f}°')
        return best

    def _set_desired_wrist_angle(self, target_pos):
        self.desired_wrist_angle = self._compute_desired_wrist_angle(target_pos)

    def _align_fingers_avoid_neighbors(self, q, target_pos):
        """Orient wrist j6 to avoid disturbing neighboring cubes.

        Uses a pre-planned wrist angle when available, so the rotation is
        chosen before motion planning and remains stable during approach.
        """
        se3 = self.ik_solver.forward_kinematics(q)

        angle = self.desired_wrist_angle
        if angle is None:
            angle = self._compute_desired_wrist_angle(target_pos)
            if angle is None:
                return self._align_fingers_tangential(q)

        return self._align_fingers_to_angle(q, angle)

    def _move_cartesian(self, target_pos, dur=None):
        """Cartesian straight-line motion with quintic time scaling.

        j6 (wrist roll) is locked to its current value so the gripper does not
        oscillate during straight-line moves.  Wrist orientation is set once
        during pre-grasp via _move_to_pose.
        """
        if self.ik_solver is None:
            return False
        if dur is None:
            dur = self._param_float('motion.slow_duration')
        if self.motion_executor is None:
            self.get_logger().error('Motion executor not ready')
            return False

        # Lock j6 so the wrist stays fixed throughout the straight-line move
        locked_j6 = float(self.commanded_arm_positions[5])
        def postprocess_q(q):
            q_out = q.copy()
            q_out[5] = locked_j6
            return q_out

        q_seed = self.commanded_arm_positions.copy()

        traj = self.motion_executor.plan_cartesian_trajectory(
            np.array(target_pos, dtype=float),
            float(dur),
            target_orientation=self.grasp_R,
            q_init=q_seed,
            max_cartesian_step=self._param_float('motion.cartesian_max_step'),
            ik_tolerance=self._param_float('ik.tolerance'),
            postprocess_q=postprocess_q,
        )
        if traj is None:
            return False
        ok = self.motion_executor.execute_trajectory_blocking(traj)
        if ok:
            self._sync_commanded_arm_state(np.array(traj.points[-1].positions, dtype=float))
        return ok

    # ---- Gripper helpers ----
    def _gripper_cmd(self, pos, effort=100.0):
        from control_msgs.action import GripperCommand
        try:
            if self.gripper_controller is None:
                return False
            if not self.gripper_controller.action_client.server_is_ready():
                return False
            g = GripperCommand.Goal()
            g.command.position = pos
            g.command.max_effort = effort
            fut = self.gripper_controller.action_client.send_goal_async(g)
            while not fut.done():
                if not self._sleep(0.02):
                    return False
            exc = fut.exception()
            if exc is not None:
                raise exc
            gh = fut.result()
            if gh is None or not gh.accepted:
                return False
            rfut = gh.get_result_async()
            deadline = time.monotonic() + self._param_float('gripper.command_timeout')
            poll_period = self._param_float('gripper.poll_period')
            pos_tol = self._param_float('gripper.position_tolerance')
            stable_eps = self._param_float('gripper.stable_epsilon')
            stable_required = self._param_int('gripper.stable_cycles')
            open_pos = self._param_float('gripper.open_position')
            finger_max_expected = self._param_float('grasp.finger_max_expected')
            stable_cycles = 0
            last_pos = float(self.current_finger_pos)

            while not rfut.done() and time.monotonic() < deadline:
                if not self._sleep(poll_period):
                    return False
                curr_pos = float(self.current_finger_pos)
                if pos <= open_pos + 1e-6:
                    if curr_pos <= pos + pos_tol:
                        break
                else:
                    if curr_pos >= finger_max_expected:
                        break
                    if abs(curr_pos - last_pos) <= stable_eps:
                        stable_cycles += 1
                        if stable_cycles >= stable_required:
                            break
                    else:
                        stable_cycles = 0
                last_pos = curr_pos

            if rfut.done():
                exc = rfut.exception()
                if exc is not None:
                    raise exc

            return self._sleep(self._param_float('gripper.wait_after_command'))
        except Exception as exc:
            self.get_logger().error(f'Gripper command failed: {exc}')
            return False

    def _open(self):
        return self._gripper_cmd(0.0)

    def _close(self):
        return self._gripper_cmd(
            self.get_parameter('gripper.close_position').value,
            self.get_parameter('gripper.max_effort').value)

    # ============================================================
    # STATE MACHINE
    # ============================================================
    def _sm_loop(self):
        self._log('SM: waiting 5 s for Gazebo + controllers...')
        if not self._sleep(5.0):
            return
        if not self._init_system():
            self.get_logger().error('Init failed!')
            return
        while rclpy.ok() and not self._stop_event.is_set():
            try:
                st = self.state
                if   st == S.INITIALIZE:   self._do_init()
                elif st == S.OBSERVE:      self._do_observe()
                elif st == S.PRE_GRASP:    self._do_pre_grasp()
                elif st == S.GRASP:        self._do_grasp()
                elif st == S.VERIFY_GRASP: self._do_verify_grasp()
                elif st == S.TRANSIT:      self._do_transit()
                elif st == S.PLACE:        self._do_place()
                elif st == S.VERIFY_PLACE: self._do_verify_place()
                elif st == S.RECOVER_DROP: self._do_recover()
                elif st == S.COMPLETE:     self._do_complete(); break
                elif st == 'IDLE':         break
                else:
                    if not self._sleep(0.1):
                        break
            except Exception as e:
                if self._should_stop():
                    break
                self.get_logger().error(f'SM error [{self.state}]: {e}')
                import traceback; self.get_logger().error(traceback.format_exc())
                if not self._sleep(1.0):
                    break

    def _init_system(self):
        # Wait for robot_description (max 10 s)
        desc = ''
        for i in range(20):
            try:
                desc = self.get_parameter('robot_description').value
                if desc: break
            except Exception: pass
            if not self._sleep(0.5):
                return False
        if not desc:
            self.get_logger().error('No robot_description!')
            return False

        # IK solver — targets tool_frame (fingertip) directly
        try:
            self.ik_solver = IKSolver(desc, 'tool_frame')
            self._log('IK solver OK (tool_frame)')
        except Exception as e:
            self.get_logger().error(f'IK init: {e}')
            return False

        self.commanded_arm_positions = self.current_arm_positions.copy()
        self.motion_executor.set_ik_solver(self.ik_solver)
        self.motion_executor.update_current_joints(self.commanded_arm_positions)

        # Grasp verifier - only 1 check required (finger gap)
        self.grasp_verifier = GraspVerifier(
            finger_min_expected=self.get_parameter('grasp.finger_min_expected').value,
            finger_max_expected=self.get_parameter('grasp.finger_max_expected').value,
            position_tolerance=self.get_parameter('grasp.position_tolerance').value,
            required_checks=self.get_parameter('grasp.required_checks').value)
        self.drop_detector = DropDetector(
            finger_min_threshold=self.get_parameter('drop.finger_min_threshold').value,
            finger_max_threshold=self.get_parameter('drop.finger_max_threshold').value)

        # Wait action servers (max 15 s each)
        if not self.motion_executor.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Arm action server timeout!')
            return False
        self._log('Arm server OK')
        if not self.gripper_controller.wait_for_server(timeout_sec=15.0):
            self.get_logger().error('Gripper server timeout!')
            return False
        self._log('Gripper server OK')

        # Wait for joint states (max 5 s)
        for _ in range(25):
            if self.latest_joint_state is not None:
                break
            if not self._sleep(0.2):
                return False
        if self.latest_joint_state is None:
            self.get_logger().error('No joint states!')
            return False

        # Quick FK verification
        for wp_name in ['home', 'observe']:
            wp = self.waypoints.get(wp_name, {})
            angles = wp.get('joint_angles')
            if angles:
                ee = self.ik_solver.forward_kinematics(np.array(angles))
                p = ee.translation
                self._log(f'FK {wp_name}: tool=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]')

        self._log('System ready (IK targets tool_frame)')
        return True

    # ---- state handlers ----
    def _do_init(self):
        home = self._home_joints()
        if not self._is_near_joint_target(home, tolerance=0.02):
            self._move_joints(home, dur=self._param_float('motion.home_duration'))
        self._transition(S.OBSERVE, 'Home reached')

    def _do_observe(self):
        obs = self._observe_joints()

        # After placing, do NOT swing to observe — just wait for fresh
        # detections from wherever the arm is.  The camera will see the
        # source row once the arm starts moving toward the next cube.
        skip_motion = (
            self.last_state in (S.VERIFY_PLACE, S.INITIALIZE) or
            (self._detections_are_fresh() and
             self.last_state == S.TRANSIT and
             self._is_near_joint_target(obs))
        )

        if not skip_motion:
            self._move_joints(obs, dur=self._param_float('motion.observe_duration'))
            # Quick settle for perception
            if not self._sleep(self._param_float('timing.observe_settle')):
                return
        else:
            # Wait until detections arrive (camera may be briefly blocked)
            deadline = time.time() + 2.0
            while not self._detections_are_fresh() and time.time() < deadline:
                if not self._sleep(0.05):
                    return

        if self.latest_cube_detections is None:
            self._log('No detections yet, waiting...')
            self._sleep(self._param_float('timing.no_detection_wait'))
            return

        if not self._detections_are_fresh():
            self._log('Perception data stale, waiting for a fresh frame...')
            self._sleep(self._param_float('timing.no_detection_wait'))
            return

        # Check if all done
        if self.pick_index >= len(self.pick_order):
            self._transition(S.COMPLETE, 'Sequence complete')
            return

        # Skip colors already fully sorted
        while self.pick_index < len(self.pick_order):
            color = self.pick_order[self.pick_index]
            target = self.cube_target_counts.get(color, 2)
            if self.cubes_sorted.get(color, 0) < target:
                break
            self.pick_index += 1
        if self.pick_index >= len(self.pick_order):
            self._transition(S.COMPLETE, 'All targets sorted')
            return

        target_color = self.pick_order[self.pick_index]
        sel = self._select_cube_by_color(target_color)
        if sel is None:
            self.no_target_count += 1
            if (
                self.no_target_count >= self.max_no_target_cycles and
                self._param_bool('selection.allow_color_fallback')
            ):
                sel = self._select_any_available_cube()
                if sel is not None:
                    self._log(f'Fallback pick: {sel[0]} (target {target_color} not visible)')
                    self.no_target_count = 0
            if sel is None:
                total = sum(self.cubes_sorted.values())
                self._log(f'No cube for {target_color} ({total} sorted)')
                self._sleep(self._param_float('timing.no_target_wait'))
                return
        else:
            self.no_target_count = 0

        color, pos = sel
        self.current_cube_color = color
        self.current_cube_pos = pos
        ci = self.cube_mapping.get(color, {})
        cp = ci.get('container_position')
        if cp:
            self.current_container_pos = np.array(cp)
        else:
            self.get_logger().error(f'No container for {color}!')
            return
        self._log(f'Target: {color} cube at {np.round(pos,3)} -> container at {cp}')
        self._transition(S.PRE_GRASP, f'{color} cube selected')

    def _select_cube_by_color(self, target_color):
        if not self.latest_cube_detections:
            return None
        cands = self._get_color_candidates(target_color)
        if not cands:
            return None
        cands.sort(key=lambda p: np.linalg.norm(p[:2]))
        self._log(f'Order [{self.pick_index}]: picking {target_color}')
        return target_color, cands[0]

    def _select_any_available_cube(self):
        if not self.latest_cube_detections:
            return None
        best = None
        for color in ['red', 'green', 'yellow']:
            target = self.cube_target_counts.get(color, 2)
            if self.cubes_sorted.get(color, 0) >= target:
                continue
            cands = self._get_color_candidates(color)
            for p in cands:
                d = np.linalg.norm(p[:2])
                if best is None or d < best[0]:
                    best = (d, color, p)
        if best is None:
            return None
        return best[1], best[2]

    def _pose_matches_color(self, pose, target_color):
        cmap = {
            'red': lambda q: q.x > 0.5,
            'green': lambda q: q.y > 0.5,
            'yellow': lambda q: q.z > 0.5,
        }
        check_fn = cmap.get(target_color)
        return False if check_fn is None else bool(check_fn(pose.orientation))

    def _workspace_limits(self):
        return (
            self._param_float('workspace.x_min'),
            self._param_float('workspace.x_max'),
            self._param_float('workspace.y_limit'),
        )

    def _is_inside_workspace(self, pos):
        x_min, x_max, y_limit = self._workspace_limits()
        return x_min <= pos[0] <= x_max and abs(pos[1]) <= y_limit

    def _tray_half_extents(self):
        margin = self._param_float('tray.verify_margin')
        half_x = 0.5 * self._param_float('tray.inner_size_x') + margin
        half_y = 0.5 * self._param_float('tray.inner_size_y') + margin
        return half_x, half_y

    def _is_in_tray_region(self, pos, color):
        color_cfg = self.cube_mapping.get(color, {})
        container = color_cfg.get('container_position')
        if container is None:
            return False
        center = np.array(container, dtype=float)
        half_x, half_y = self._tray_half_extents()
        return (
            abs(pos[0] - center[0]) <= half_x and
            abs(pos[1] - center[1]) <= half_y
        )

    def _is_in_any_tray(self, pos):
        for color in ['red', 'green', 'yellow']:
            if self._is_in_tray_region(pos, color):
                return True
        return False

    def _get_color_detections(self, target_color):
        if not self._detections_are_fresh():
            return []
        detections = []
        for pose in self.latest_cube_detections.poses:
            if not self._pose_matches_color(pose, target_color):
                continue
            pos = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
            if self._is_inside_workspace(pos):
                detections.append(pos)
        return detections

    def _get_color_candidates(self, target_color):
        detections = self._get_color_detections(target_color)
        return [p for p in detections if not self._is_in_any_tray(p)]

    def _refresh_current_cube_position(self):
        if self.current_cube_color is None:
            return
        cands = self._get_color_candidates(self.current_cube_color)
        if not cands:
            return
        if self.current_cube_pos is None:
            cands.sort(key=lambda p: np.linalg.norm(p[:2]))
            self.current_cube_pos = cands[0]
        else:
            cands.sort(key=lambda p: np.linalg.norm((p - self.current_cube_pos)[:2]))
            nearest = cands[0]
            nearest_dist = float(np.linalg.norm((nearest - self.current_cube_pos)[:2]))
            if nearest_dist <= self._param_float('selection.max_retarget_distance'):
                self.current_cube_pos = nearest
            else:
                self._log(
                    f'Keep {self.current_cube_color} target: nearest visible cube is '
                    f'{nearest_dist:.3f}m away from locked target'
                )
        self._log(f'Refresh {self.current_cube_color} target: {np.round(self.current_cube_pos,3)}')

    def _compute_place_slot_offset(self, slot_index, num_slots):
        axis = str(self.get_parameter('placement.slot_axis').value).strip().lower()
        cube_size = self._param_float('cube.size')
        wall_clearance = self._param_float('tray.wall_clearance')
        tray_inner_x = self._param_float('tray.inner_size_x')
        tray_inner_y = self._param_float('tray.inner_size_y')

        offset = np.zeros(3, dtype=float)
        if num_slots <= 1:
            return offset

        if axis == 'x':
            tray_length = tray_inner_x
            axis_index = 0
        else:
            tray_length = tray_inner_y
            axis_index = 1

        usable_span = tray_length - cube_size - 2.0 * wall_clearance
        usable_span = max(0.0, usable_span)
        if axis == 'x':
            slot_centers = np.linspace(0.5 * usable_span, -0.5 * usable_span, num_slots)
        else:
            slot_centers = np.linspace(-0.5 * usable_span, 0.5 * usable_span, num_slots)
        offset[axis_index] = slot_centers[min(slot_index, num_slots - 1)]
        return offset

    def _get_place_target(self):
        """Return the selected tray slot computed from tray geometry."""
        if self.current_cube_color is None or self.current_container_pos is None:
            return None

        color_cfg = self.cube_mapping.get(self.current_cube_color, {})
        num_slots = int(color_cfg.get('num_cubes', 2))
        slot_index = self.cubes_sorted.get(self.current_cube_color, 0)
        slot = self.current_container_pos + self._compute_place_slot_offset(slot_index, num_slots)

        self._log(
            f'Place slot {self.current_cube_color}[{slot_index}] -> {np.round(slot, 3)}'
        )
        return slot

    def _is_in_slot_region(self, pos, slot_center):
        if slot_center is None:
            return False
        slot_center = np.array(slot_center, dtype=float)
        half_extent = 0.5 * self._param_float('cube.size') + self._param_float('tray.verify_margin')
        return (
            abs(pos[0] - slot_center[0]) <= half_extent and
            abs(pos[1] - slot_center[1]) <= half_extent
        )

    def _count_cubes_in_tray(self, color):
        return sum(
            1 for pos in self._get_color_detections(color)
            if self._is_in_tray_region(pos, color)
        )

    def _wait_for_place_confirmation(self, color, baseline_count, slot_target=None):
        timeout = self._param_float('placement.verify_timeout')
        poll = self._param_float('placement.verify_poll_period')
        deadline = time.time() + timeout
        best_count = baseline_count

        while time.time() < deadline:
            if not self._detections_are_fresh(timeout=max(poll * 2.0, 0.2)):
                if not self._sleep(poll):
                    break
                continue

            detections = self._get_color_detections(color)
            count = sum(1 for pos in detections if self._is_in_tray_region(pos, color))
            best_count = max(best_count, count)
            if slot_target is not None and any(self._is_in_slot_region(pos, slot_target) for pos in detections):
                return True, best_count
            if count >= baseline_count + 1:
                return True, count
            if not self._sleep(poll):
                break

        return False, best_count

    def _do_pre_grasp(self):
        self._refresh_current_cube_position()
        if self.current_cube_pos is not None:
            self._set_desired_wrist_angle(self.current_cube_pos)
        self._open()

        off = self.get_parameter('offsets.pre_grasp_z').value
        pos = self.current_cube_pos.copy()
        pos[2] += off

        # If the arm is still over the tray area (x > 0.28), plan a smooth
        # multi-waypoint trajectory: mid-retract → above target → pre-grasp
        # with non-zero intermediate velocities (no halting).
        tool = self._tool_translation()
        if tool is not None and tool[0] > 0.28:
            retract_z = self._param_float('motion.transit_clearance_z')
            mid_pos = np.array([0.25, 0.0, retract_z])
            above_target = np.array([pos[0], pos[1], retract_z])
            self._log(f'Smooth retract: mid {np.round(mid_pos,3)} '
                       f'-> above {np.round(above_target,3)} -> pre {np.round(pos,3)}')
            ok = self._move_through_poses(
                [mid_pos, above_target, pos],
                [0.30, 0.30, 0.25],
            )
            if ok:
                self._transition(S.GRASP, 'Above cube')
                return
            self._log('Smooth retract failed, fallback to direct')

        self._log(f'Pre-grasp: {np.round(pos,3)}')
        ok = self._move_to_pose(pos, dur=self._param_float('motion.pre_grasp_duration'))
        if ok:
            self._transition(S.GRASP, 'Above cube')
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.retry_count = 0
                self._transition(S.OBSERVE, 'Pre-grasp failed')
            else:
                self._log(f'Pre-grasp retry {self.retry_count}/{self.max_retries}')

    def _do_grasp(self):
        """Cartesian descent, close, check, lift — then straight to TRANSIT."""
        off = self.get_parameter('offsets.grasp_z').value
        pos = self.current_cube_pos.copy()
        pos[2] += off
        self._log(f'Grasp: {np.round(pos,3)}')

        # Straight-down Cartesian descent (j6 locked by _move_cartesian)
        ok = self._move_cartesian(
            pos, dur=self._param_float('motion.grasp_cartesian_duration')
        )
        if not ok:
            ok = self._move_to_pose(pos, dur=0.40, strict=False)
            if not ok:
                self._transition(S.OBSERVE, 'Grasp descent fail')
                return

        self._close()
        if not self._sleep(0.10):
            return

        # Check finger gap BEFORE lifting
        finger_ok = self.grasp_verifier.check_finger_gap(self.current_finger_pos)
        self._log(f'Grasp check: finger={self.current_finger_pos:.3f}, ok={finger_ok}')
        if not finger_ok:
            self.retry_count += 1
            self._open()
            if self.retry_count >= self.max_retries:
                self.retry_count = 0
                self._transition(S.OBSERVE, f'Grasp failed {self.max_retries}x')
            else:
                self._transition(S.PRE_GRASP, 'Retry grasp')
            return

        # Lift (Cartesian, straight up)
        lift_pos = self.current_cube_pos.copy()
        lift_pos[2] += self._param_float('offsets.post_grasp_z')
        ok = self._move_cartesian(lift_pos, dur=0.20)
        if not ok:
            self._move_to_pose(lift_pos, dur=0.25)

        # Re-check after lift
        finger_ok = self.grasp_verifier.check_finger_gap(self.current_finger_pos)
        if finger_ok:
            self.retry_count = 0
            self._transition(S.TRANSIT, 'Grasp verified')
        else:
            self._open()
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.retry_count = 0
                self._transition(S.OBSERVE, 'Lost during lift')
            else:
                self._transition(S.PRE_GRASP, 'Retry')

    def _do_verify_grasp(self):
        # Normally skipped — GRASP goes directly to TRANSIT.
        # This is a safety catch in case we enter this state.
        finger_ok = self.grasp_verifier.check_finger_gap(self.current_finger_pos)
        if finger_ok:
            self.retry_count = 0
            self._transition(S.TRANSIT, 'Grasp OK')
        else:
            self._open()
            self._transition(S.OBSERVE, 'Grasp lost')

    def _do_transit(self):
        place_target = self._get_place_target()
        if place_target is None:
            self._transition(S.RECOVER_DROP, 'No place target')
            return

        self.current_cube_pos = None
        self.desired_wrist_angle = None

        tool = self._tool_translation()
        if tool is None:
            self._transition(S.RECOVER_DROP, 'No FK')
            return

        z_clear = self._param_float('motion.transit_clearance_z')

        # Step 1: Lift straight up to clearance (Cartesian — keeps cube stable)
        if float(tool[2]) < z_clear - 0.01:
            lift_pos = tool.copy()
            lift_pos[2] = z_clear
            ok = self._move_cartesian(lift_pos, dur=0.25)
            if not ok:
                ok = self._move_to_pose(
                    [float(tool[0]), float(tool[1]), z_clear], dur=0.30, strict=False)
            if not ok:
                self._transition(S.RECOVER_DROP, 'Lift fail')
                return

        # Step 2: Single joint-space move above tray at clearance height
        above_tray = place_target.copy()
        above_tray[2] = z_clear
        self._log(f'Transit to {np.round(above_tray, 3)}')
        ok = self._move_to_pose(above_tray, dur=0.45, strict=False)
        if not ok:
            self._transition(S.RECOVER_DROP, 'Transit fail')
            return

        if self.drop_detector.check_for_drop(self.current_finger_pos):
            self._transition(S.RECOVER_DROP, 'Drop during transit')
            return

        self._transition(S.PLACE, 'Above tray')

    def _do_place(self):
        """Drop cube by opening gripper from current height."""
        self._log(f'Dropping {self.current_cube_color}')
        self._open()
        self._transition(S.VERIFY_PLACE, f'Dropped {self.current_cube_color}')

    def _do_verify_place(self):
        """Count the cube as placed and advance.

        We drop from directly above the container — verification via camera
        is unreliable because the arm blocks the view.  Trust the drop.
        """
        placed_color = self.current_cube_color
        baseline_count = self.cubes_sorted.get(placed_color, 0)

        self.cubes_sorted[placed_color] = baseline_count + 1
        self.retry_count = 0
        self._clear_active_target()

        # Advance pick_index past any fully-sorted colors
        while self.pick_index < len(self.pick_order):
            color = self.pick_order[self.pick_index]
            target = self.cube_target_counts.get(color, 2)
            if self.cubes_sorted.get(color, 0) < target:
                break
            self.pick_index += 1

        total = sum(self.cubes_sorted.values())
        self._log(f'{placed_color} sorted! {total}/{self.total_cubes} {self.cubes_sorted}')
        if self.pick_index >= len(self.pick_order):
            self._transition(S.COMPLETE, 'All done')
        else:
            self._transition(S.OBSERVE, f'Next: {self.pick_order[self.pick_index]}')

    def _do_recover(self):
        self._log('Recovery...')
        self._open()
        home = self.waypoints.get('home', {}).get('joint_angles', [0.0]*6)
        self._move_joints(home, dur=self._param_float('motion.recover_home_duration'))
        if not self._sleep(self._param_float('timing.recover_settle')):
            return
        self._clear_active_target(reset_color=True)
        self._transition(S.OBSERVE, 'Recovered')

    def _do_complete(self):
        home = self.waypoints.get('home', {}).get('joint_angles', [0.0]*6)
        self._move_joints(home, dur=self._param_float('motion.complete_home_duration'))
        self._log('='*50)
        self._log(f'COMPLETE — sorted {self.cubes_sorted}')
        self._log('='*50)
        self.state = 'IDLE'


def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    exe = MultiThreadedExecutor(num_threads=4)
    exe.add_node(node)
    try:
        exe.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            exe.shutdown()
        except Exception:
            pass
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
