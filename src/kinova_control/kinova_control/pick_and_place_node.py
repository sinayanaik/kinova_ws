#!/usr/bin/env python3
"""
pick_and_place_node.py  –  FAST state machine for Kinova Gen3 Lite pick-and-place.

States: INITIALIZE → OBSERVE → PRE_GRASP → GRASP → VERIFY_GRASP →
        TRANSIT_TO_PLACE → PLACE → VERIFY_PLACE → COMPLETE
        (with RECOVER_DROP fallback)

Design choices for speed:
 - Short trajectory durations (1-2 s instead of 3-4 s).
 - Minimal sleep() calls.  Only wait where physics needs settling.
 - Joint-space interpolation with few waypoints for large motions.
 - Cartesian interpolation only for final descent/ascent (~6 cm).
 - Grasp verification requires only 1 check (finger gap).
 - No IK warm-up tests during initialization.
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

        # Ordered picking: red -> green -> yellow -> red -> green -> yellow
        self.pick_order = ['red', 'green', 'yellow', 'red', 'green', 'yellow']
        self.pick_index = 0

        # ---------- Sensor data ----------
        self.latest_joint_state = None
        self.latest_cube_detections = None
        self.arm_joint_names = None
        self.current_arm_positions = np.zeros(6)
        self.current_finger_pos = 0.0

        # Load configs
        self._load_configs()

        # Subscribers
        self.create_subscription(
            JointState, '/joint_states',
            self._joint_state_cb, 10, callback_group=self.cb_group)
        self.create_subscription(
            PoseArray, '/perception/cube_detections',
            self._det_cb, 10, callback_group=self.cb_group)

        # URDF for Pinocchio
        self.declare_parameter('robot_description', '')
        self.ik_solver = None
        self.gripper_controller = None
        self.grasp_verifier = None
        self.drop_detector = None

        # Action client
        self.arm_ac = ActionClient(
            self, FollowJointTrajectory,
            '/joint_trajectory_controller/follow_joint_trajectory')

        # Grasp orientation: tool_frame pointing straight down
        # tool_frame has Rz(π/2) offset from end_effector_link
        self.grasp_R = np.array([
            [ 0.0, -1.0,  0.0],
            [-1.0,  0.0,  0.0],
            [ 0.0,  0.0, -1.0]])

        # Start state-machine thread
        threading.Thread(target=self._sm_loop, daemon=True).start()

    # ============================================================
    # PARAMETERS
    # ============================================================
    def _declare_params(self):
        p = self.declare_parameter
        p('ik.max_iterations', 300)
        p('ik.tolerance', 0.002)
        p('ik.damping_lambda', 1e-4)
        p('ik.step_size', 0.5)
        p('motion.interpolation_steps', 8)
        p('motion.waypoint_duration', 0.8)
        p('motion.slow_duration', 0.8)
        p('motion.transit_height', 0.35)
        p('gripper.open_position', 0.0)
        p('gripper.close_position', 0.85)
        p('gripper.max_effort', 100.0)
        p('gripper.wait_after_command', 0.3)
        p('grasp.finger_min_expected', 0.15)
        p('grasp.finger_max_expected', 0.65)
        p('grasp.position_tolerance', 0.05)
        p('grasp.required_checks', 1)
        p('drop.finger_min_threshold', 0.1)
        p('drop.finger_max_threshold', 0.80)
        p('offsets.pre_grasp_z', 0.05)
        p('offsets.grasp_z', -0.005)
        p('offsets.post_grasp_z', 0.08)
        p('offsets.pre_place_z', 0.10)
        p('offsets.drop_height', 0.07)  # height above container to open gripper and drop
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

    def _log(self, txt):
        self.get_logger().info(txt)

    def _transition(self, new, reason):
        self._log(f'STATE {self.state} -> {new} | {reason}')
        self.state = new

    # ============================================================
    # MOTION HELPERS  (keep durations SHORT)
    # ============================================================
    def _send_traj(self, wps, dur):
        """Send joint trajectory blocking. wps = list of 6-arrays."""
        if not self.arm_ac.server_is_ready():
            self.get_logger().error('Arm action server not ready!')
            return False
        traj = JointTrajectory()
        traj.joint_names = self.arm_joint_names
        n = len(wps)
        dt = dur / max(n, 1)
        for i, q in enumerate(wps):
            pt = JointTrajectoryPoint()
            pt.positions = list(q)
            if i == n - 1:
                pt.velocities = [0.0] * 6
            t = (i + 1) * dt
            pt.time_from_start = Duration(sec=int(t), nanosec=int((t % 1) * 1e9))
            traj.points.append(pt)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = traj
        fut = self.arm_ac.send_goal_async(goal)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Trajectory REJECTED')
            return False
        rfut = gh.get_result_async()
        while not rfut.done():
            time.sleep(0.02)
        return True

    def _move_joints(self, target, dur=1.5):
        """Joint-space interpolation (5 waypoints)."""
        cur = self.current_arm_positions.copy()
        tgt = np.array(target)
        wps = [cur + float(i)/5 * (tgt - cur) for i in range(1, 6)]
        return self._send_traj(wps, dur)

    def _move_to_pose(self, pos, dur=1.5):
        """IK + joint-space move. Uses solve_robust (6D → 5D Z-down → pos-only)."""
        if self.ik_solver is None:
            return False
        ok, q = self.ik_solver.solve_robust(
            np.array(pos), self.grasp_R,
            self.current_arm_positions.copy(), 300, 2e-3)
        ee = self.ik_solver.forward_kinematics(q)
        err = np.linalg.norm(ee.translation - np.array(pos))

        # Check orientation: tool Z should point down
        tool_z = ee.rotation[:, 2]
        z_err = np.linalg.norm(tool_z - np.array([0, 0, -1]))

        if err > 0.008:
            # Try Z-axis-down fallback (5D: position + vertical orientation)
            self._log(f'Robust IK err={err:.4f}, trying Z-down 5D...')
            ok2, q2 = self.ik_solver.solve_z_axis_down(
                np.array(pos),
                self.current_arm_positions.copy(), 500, 2e-3)
            ee2 = self.ik_solver.forward_kinematics(q2)
            err2 = np.linalg.norm(ee2.translation - np.array(pos))
            z_err2 = np.linalg.norm(ee2.rotation[:, 2] - np.array([0, 0, -1]))
            if err2 < err:
                q, ee, err, z_err = q2, ee2, err2, z_err2
            if err > 0.01:
                self.get_logger().warn(
                    f'IK FAIL target={np.round(pos,3)} '
                    f'achieved={np.round(ee.translation,3)} err={err:.4f} z_err={z_err:.4f}')
                return False
        self._log(f'IK OK target={np.round(pos,3)} '
                  f'achieved={np.round(ee.translation,3)} err={err:.4f} z_err={z_err:.4f}')
        return self._move_joints(q, dur)

    def _move_cartesian(self, target_pos, dur=1.5):
        """Short Cartesian interpolation (fine motion, preserves vertical orientation)."""
        if self.ik_solver is None:
            return False
        ce = self.ik_solver.forward_kinematics(self.current_arm_positions)
        sp = ce.translation.copy()
        tp = np.array(target_pos)
        dist = np.linalg.norm(tp - sp)
        n = max(3, min(int(dist / 0.015), 10))
        qp = self.current_arm_positions.copy()
        wps = []
        for i in range(1, n + 1):
            t = i / float(n)
            p = sp + t * (tp - sp)
            # Try 6D first, then Z-down 5D fallback
            ok, q = self.ik_solver.solve(p, self.grasp_R, qp, 200, 2e-3, 1e-4)
            if not ok:
                ok, q = self.ik_solver.solve_z_axis_down(p, qp, 200, 3e-3)
                if not ok:
                    self.get_logger().warn(f'Cart IK fail step {i}/{n}')
                    return False
            wps.append(q)
            qp = q.copy()
        return self._send_traj(wps, dur)

    # ---- Gripper helpers ----
    def _gripper_cmd(self, pos, effort=100.0):
        from control_msgs.action import GripperCommand
        if self.gripper_controller is None:
            return False
        if not self.gripper_controller.action_client.server_is_ready():
            return False
        g = GripperCommand.Goal()
        g.command.position = pos
        g.command.max_effort = effort
        fut = self.gripper_controller.action_client.send_goal_async(g)
        while not fut.done():
            time.sleep(0.02)
        gh = fut.result()
        if not gh.accepted:
            return False
        rfut = gh.get_result_async()
        while not rfut.done():
            time.sleep(0.02)
        time.sleep(0.25)
        return True

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
        time.sleep(5.0)
        if not self._init_system():
            self.get_logger().error('Init failed!')
            return
        while rclpy.ok():
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
                else:                      time.sleep(0.1)
            except Exception as e:
                self.get_logger().error(f'SM error [{self.state}]: {e}')
                import traceback; self.get_logger().error(traceback.format_exc())
                time.sleep(1.0)

    def _init_system(self):
        # Wait for robot_description (max 10 s)
        desc = ''
        for i in range(20):
            try:
                desc = self.get_parameter('robot_description').value
                if desc: break
            except Exception: pass
            time.sleep(0.5)
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

        # Gripper
        self.gripper_controller = GripperController(
            node=self,
            open_position=self.get_parameter('gripper.open_position').value,
            close_position=self.get_parameter('gripper.close_position').value,
            max_effort=self.get_parameter('gripper.max_effort').value,
            wait_after_command=self.get_parameter('gripper.wait_after_command').value)

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
        if not self.arm_ac.wait_for_server(timeout_sec=15.0):
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
            time.sleep(0.2)
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

        self._log('System ready (IK targets tool_frame = fingertip)')
        return True

    # ---- state handlers ----
    def _do_init(self):
        home = self.waypoints.get('home', {}).get(
            'joint_angles', [0.0]*6)
        self._move_joints(home, dur=1.2)
        self._transition(S.OBSERVE, 'Home reached')

    def _do_observe(self):
        obs = self.waypoints.get('observe', {}).get(
            'joint_angles', [0.0, 0.35, 1.57, 0.0, -1.05, 0.0])
        self._move_joints(obs, dur=1.0)
        # Quick settle for perception
        time.sleep(0.5)

        if self.latest_cube_detections is None:
            self._log('No detections yet, waiting...')
            time.sleep(1.0)
            return

        # Check if all done
        if self.pick_index >= len(self.pick_order):
            self._transition(S.COMPLETE, 'Sequence complete')
            return

        sel = self._select_cube()
        if sel is None:
            total = sum(self.cubes_sorted.values())
            self._log(f'No cube for {self.pick_order[self.pick_index]} ({total} sorted)')
            time.sleep(1.0)
            return

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

    def _select_cube(self):
        """Select the next cube following R->G->Y->R->G->Y order."""
        if not self.latest_cube_detections:
            return None
        cmap = {
            'red':    lambda q: q.x > 0.5,
            'green':  lambda q: q.y > 0.5,
            'yellow': lambda q: q.z > 0.5}

        if self.pick_index >= len(self.pick_order):
            return None

        target_color = self.pick_order[self.pick_index]
        check_fn = cmap[target_color]

        # Find all cubes of the target color
        cands = []
        for pose in self.latest_cube_detections.poses:
            if check_fn(pose.orientation):
                p = np.array([pose.position.x, pose.position.y, pose.position.z])
                cands.append(p)

        if not cands:
            self._log(f'No {target_color} cube found, skipping')
            self.pick_index += 1
            return None

        # Pick the closest one
        cands.sort(key=lambda p: np.linalg.norm(p[:2]))
        self._log(f'Order [{self.pick_index}]: picking {target_color}')
        return target_color, cands[0]

    def _do_pre_grasp(self):
        self._open()
        off = self.get_parameter('offsets.pre_grasp_z').value
        pos = self.current_cube_pos.copy()
        pos[2] += off
        self._log(f'Pre-grasp tool target: {np.round(pos,3)} (fingers at z={pos[2]:.3f})')
        ok = self._move_to_pose(pos, dur=0.8)
        if ok:
            self._transition(S.GRASP, 'Above cube')
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.pick_index += 1  # skip this cube in sequence
                self.retry_count = 0
                self._transition(S.OBSERVE, 'Max retries, skip cube')
            else:
                # Stay close — try again without retreating to observe
                self._log(f'Pre-grasp retry {self.retry_count}/{self.max_retries}')
                time.sleep(0.3)

    def _do_grasp(self):
        off = self.get_parameter('offsets.grasp_z').value
        pos = self.current_cube_pos.copy()
        pos[2] += off
        self._log(f'Grasp tool target: {np.round(pos,3)} (fingers at z={pos[2]:.3f})')
        ok = self._move_cartesian(pos, dur=0.7)
        if not ok:
            ok = self._move_to_pose(pos, dur=0.7)
            if not ok:
                self._transition(S.OBSERVE, 'Grasp descent fail')
                return
        time.sleep(0.3)  # settle before closing
        self._close()
        time.sleep(0.5)  # wait for gripper to fully close/stall

        # Immediate grasp check BEFORE lifting
        finger_ok = self.grasp_verifier.check_finger_gap(self.current_finger_pos)
        self._log(f'Immediate grasp check: finger={self.current_finger_pos:.3f}, '
                  f'range=[{self.grasp_verifier.finger_min:.2f},{self.grasp_verifier.finger_max:.2f}], '
                  f'ok={finger_ok}')
        if not finger_ok:
            # Don't even lift — open and retry immediately
            self.retry_count += 1
            self._open()
            if self.retry_count >= self.max_retries:
                self.pick_index += 1
                self.retry_count = 0
                self._transition(S.OBSERVE, f'Grasp failed {self.max_retries}x, skip')
            else:
                self._log(f'Empty grasp, retry {self.retry_count}/{self.max_retries}')
                self._transition(S.PRE_GRASP, 'Retry grasp')
            return

        self._transition(S.VERIFY_GRASP, 'Cube in gripper')

    def _do_verify_grasp(self):
        # Lift up
        off = self.get_parameter('offsets.post_grasp_z').value
        pos = self.current_cube_pos.copy()
        pos[2] += off
        ok = self._move_cartesian(pos, dur=0.7)
        if not ok:
            self._move_to_pose(pos, dur=0.7)
        time.sleep(0.2)

        # Check finger gap only (required_checks=1)
        finger_ok = self.grasp_verifier.check_finger_gap(self.current_finger_pos)
        self._log(f'Grasp check: finger_pos={self.current_finger_pos:.3f}, ok={finger_ok}')

        if finger_ok:
            self.retry_count = 0
            self._transition(S.TRANSIT, 'Grasp OK')
        else:
            self.retry_count += 1
            if self.retry_count >= self.max_retries:
                self.pick_index += 1  # skip this cube in sequence
                self.retry_count = 0
                self._open()
                self._transition(S.OBSERVE, 'Max retries, skip')
            else:
                # Stay close — retry from pre-grasp instead of retreating
                self._open()
                self._log(f'Grasp fail retry {self.retry_count}/{self.max_retries}, re-approaching')
                self._transition(S.PRE_GRASP, 'Retry from pre-grasp')

    def _do_transit(self):
        cp = self.current_container_pos.copy()
        drop_h = self.get_parameter('offsets.drop_height').value
        above = np.array([cp[0], cp[1], cp[2] + drop_h])
        self._log(f'Transit to drop pos {np.round(above,3)}')
        ok = self._move_to_pose(above, dur=0.8)
        if self.drop_detector.check_for_drop(self.current_finger_pos):
            self._transition(S.RECOVER_DROP, 'Drop during transit')
            return
        if not ok:
            self._transition(S.RECOVER_DROP, 'Transit fail')
            return
        self._transition(S.PLACE, 'Above container, ready to drop')

    def _do_place(self):
        """Simply open gripper to drop the cube into the container."""
        cp = self.current_container_pos.copy()
        drop_h = self.get_parameter('offsets.drop_height').value
        drop_pos = np.array([cp[0], cp[1], cp[2] + drop_h])
        self._log(f'Dropping cube at {np.round(drop_pos,3)}')
        # Open gripper — cube falls into container
        self._open()
        time.sleep(0.3)  # wait for cube to settle
        # Retreat upward
        retreat_z = cp[2] + self.get_parameter('offsets.pre_place_z').value
        self._move_to_pose([cp[0], cp[1], retreat_z], dur=0.6)
        self._transition(S.VERIFY_PLACE, f'Dropped {self.current_cube_color}')

    def _do_verify_place(self):
        time.sleep(0.3)
        self.cubes_sorted[self.current_cube_color] = \
            self.cubes_sorted.get(self.current_cube_color, 0) + 1
        self.retry_count = 0
        self.pick_index += 1
        total = sum(self.cubes_sorted.values())
        self._log(f'{self.current_cube_color} sorted! {total}/{self.total_cubes} {self.cubes_sorted}')
        if self.pick_index >= len(self.pick_order):
            self._transition(S.COMPLETE, 'All done')
        else:
            self._transition(S.OBSERVE, f'Next: {self.pick_order[self.pick_index]}')

    def _do_recover(self):
        self._log('Recovery...')
        self._open()
        home = self.waypoints.get('home', {}).get('joint_angles', [0.0]*6)
        self._move_joints(home, dur=1.5)
        time.sleep(0.5)
        self._transition(S.OBSERVE, 'Recovered')

    def _do_complete(self):
        home = self.waypoints.get('home', {}).get('joint_angles', [0.0]*6)
        self._move_joints(home, dur=2.0)
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
        node.destroy_node()
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
