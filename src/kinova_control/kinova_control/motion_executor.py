#!/usr/bin/env python3
"""
motion_executor.py
Trajectory generation and execution for the Kinova Gen3 Lite.

The planner uses quintic time scaling so every motion starts and ends with
zero velocity and zero acceleration. Joint-space motions are generated from
the closed-form quintic polynomial. Cartesian motions use the same quintic
path parameter along either a straight-line task-space segment or a continuous
polyline parameterized by cumulative arc length.
"""
import time
from typing import Callable, Optional

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from kinova_control.ik_solver import IKSolver


class MotionExecutor:
    """Generate and execute quintic trajectories for the arm."""

    def __init__(
        self,
        node: Node,
        ik_solver: Optional[IKSolver],
        arm_joint_names,
        action_name: str = '/joint_trajectory_controller/follow_joint_trajectory',
        sample_period: float = 0.02,
        min_samples: int = 16,
        max_cartesian_step: float = 0.008,
        joint_velocity_limit: float = 0.9,
        joint_acceleration_limit: float = 2.0,
    ):
        self.node = node
        self.ik_solver = ik_solver
        self.joint_names = list(arm_joint_names)
        self.sample_period = max(0.01, float(sample_period))
        self.min_samples = max(4, int(min_samples))
        self.max_cartesian_step = max(0.002, float(max_cartesian_step))
        self.joint_velocity_limit = max(0.1, float(joint_velocity_limit))
        self.joint_acceleration_limit = max(0.1, float(joint_acceleration_limit))
        self.current_joints = np.zeros(len(self.joint_names), dtype=float)
        self.action_client = ActionClient(
            node, FollowJointTrajectory, action_name
        )

    def set_ik_solver(self, ik_solver: IKSolver):
        self.ik_solver = ik_solver

    def wait_for_server(self, timeout_sec: float = 30.0) -> bool:
        return self.action_client.wait_for_server(timeout_sec=timeout_sec)

    def update_current_joints(self, joint_positions: np.ndarray):
        self.current_joints = np.array(joint_positions, dtype=float).copy()

    def get_current_joints(self) -> np.ndarray:
        return self.current_joints.copy()

    @staticmethod
    def _duration_msg(seconds: float) -> Duration:
        seconds = max(0.0, float(seconds))
        sec = int(seconds)
        nanosec = int(round((seconds - sec) * 1e9))
        if nanosec >= int(1e9):
            sec += 1
            nanosec -= int(1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def _sample_quintic_profile(
        self,
        duration: float,
        min_samples: Optional[int] = None,
    ):
        min_samples = self.min_samples if min_samples is None else int(min_samples)
        min_samples = max(4, min_samples)
        duration = max(float(duration), self.sample_period * (min_samples - 1))

        sample_count = max(
            min_samples,
            int(np.ceil(duration / self.sample_period)) + 1,
        )
        times = np.linspace(0.0, duration, sample_count)
        tau = times / duration

        sigma = 10.0 * tau**3 - 15.0 * tau**4 + 6.0 * tau**5
        sigma_dot = (
            30.0 * tau**2 - 60.0 * tau**3 + 30.0 * tau**4
        ) / duration
        sigma_ddot = (
            60.0 * tau - 180.0 * tau**2 + 120.0 * tau**3
        ) / (duration**2)

        sigma_dot[0] = 0.0
        sigma_dot[-1] = 0.0
        sigma_ddot[0] = 0.0
        sigma_ddot[-1] = 0.0
        return duration, times, sigma, sigma_dot, sigma_ddot

    def _trajectory_from_samples(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
        times: np.ndarray,
    ) -> JointTrajectory:
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        for idx in range(1, len(times)):
            point = JointTrajectoryPoint()
            point.positions = positions[idx].tolist()
            point.velocities = velocities[idx].tolist()
            point.accelerations = accelerations[idx].tolist()
            point.time_from_start = self._duration_msg(times[idx])
            trajectory.points.append(point)

        return trajectory

    def _limit_time_scaling(
        self,
        times: np.ndarray,
        velocities: np.ndarray,
        accelerations: np.ndarray,
    ):
        peak_velocity = float(np.max(np.abs(velocities))) if velocities.size else 0.0
        peak_acceleration = float(np.max(np.abs(accelerations))) if accelerations.size else 0.0

        vel_scale = peak_velocity / self.joint_velocity_limit if self.joint_velocity_limit > 0.0 else 1.0
        acc_scale = np.sqrt(
            peak_acceleration / self.joint_acceleration_limit
        ) if self.joint_acceleration_limit > 0.0 and peak_acceleration > 0.0 else 1.0
        scale = max(1.0, vel_scale, acc_scale)
        if scale <= 1.0:
            return times, 1.0

        duration = times[-1] * scale
        scaled_times = np.linspace(0.0, duration, len(times))
        return scaled_times, scale

    @staticmethod
    def _numerical_derivatives(samples: np.ndarray, times: np.ndarray):
        velocities = np.zeros_like(samples)
        accelerations = np.zeros_like(samples)

        if len(samples) < 2:
            return velocities, accelerations

        for idx in range(1, len(samples) - 1):
            dt = times[idx + 1] - times[idx - 1]
            velocities[idx] = (samples[idx + 1] - samples[idx - 1]) / max(dt, 1e-6)

        for idx in range(1, len(samples) - 1):
            dt_prev = times[idx] - times[idx - 1]
            dt_next = times[idx + 1] - times[idx]
            v_prev = (samples[idx] - samples[idx - 1]) / max(dt_prev, 1e-6)
            v_next = (samples[idx + 1] - samples[idx]) / max(dt_next, 1e-6)
            accelerations[idx] = 2.0 * (v_next - v_prev) / max(dt_prev + dt_next, 1e-6)

        velocities[0] = 0.0
        velocities[-1] = 0.0
        accelerations[0] = 0.0
        accelerations[-1] = 0.0
        return velocities, accelerations

    def _wait_for_future(self, future, poll_period: float = 0.01):
        while not future.done():
            if not rclpy.ok():
                return None
            time.sleep(poll_period)

        exc = future.exception()
        if exc is not None:
            raise exc
        return future.result()

    def plan_joint_trajectory(
        self,
        target_joints: np.ndarray,
        duration: float,
        q_init: Optional[np.ndarray] = None,
    ) -> JointTrajectory:
        q_start = self.current_joints.copy() if q_init is None else np.array(q_init, dtype=float)
        q_goal = np.array(target_joints, dtype=float)

        _, times, sigma, sigma_dot, sigma_ddot = self._sample_quintic_profile(duration)
        delta = q_goal - q_start

        positions = q_start + np.outer(sigma, delta)
        velocities = np.outer(sigma_dot, delta)
        accelerations = np.outer(sigma_ddot, delta)

        scaled_times, scale = self._limit_time_scaling(times, velocities, accelerations)
        if scale > 1.0:
            _, scaled_times, sigma, sigma_dot, sigma_ddot = self._sample_quintic_profile(
                float(scaled_times[-1]),
                min_samples=len(times),
            )
            positions = q_start + np.outer(sigma, delta)
            velocities = np.outer(sigma_dot, delta)
            accelerations = np.outer(sigma_ddot, delta)
            times = scaled_times

        return self._trajectory_from_samples(positions, velocities, accelerations, times)

    def plan_multi_waypoint_trajectory(
        self,
        waypoint_joints: list,
        durations: list,
        q_init: Optional[np.ndarray] = None,
    ) -> JointTrajectory:
        """Plan a smooth trajectory through multiple joint-space waypoints.

        Uses cubic Hermite interpolation with estimated intermediate
        velocities so the robot passes through each waypoint without
        stopping.  Only the very first and very last points have zero
        velocity (quintic ease-in / ease-out at the endpoints).
        """
        q_start = self.current_joints.copy() if q_init is None else np.array(q_init, dtype=float)
        wps = [q_start.copy()] + [np.array(w, dtype=float) for w in waypoint_joints]
        n_seg = len(wps) - 1
        if n_seg < 1:
            return self.plan_joint_trajectory(q_start, 0.1)

        # Pad durations
        durs = [max(float(d), self.sample_period * 4) for d in durations]
        while len(durs) < n_seg:
            durs.append(durs[-1])
        durs = durs[:n_seg]

        # Estimate intermediate velocities using Catmull-Rom style:
        # v_i = (q_{i+1} - q_{i-1}) / (T_{i-1} + T_i)
        # Boundary: v_0 = 0, v_n = 0
        vels = [np.zeros_like(q_start)] * (n_seg + 1)
        for i in range(1, n_seg):
            dt = durs[i - 1] + durs[i]
            vels[i] = (wps[i + 1] - wps[i - 1]) / max(dt, 1e-6)
        vels[0] = np.zeros_like(q_start)
        vels[n_seg] = np.zeros_like(q_start)

        # Build concatenated samples via cubic Hermite per segment
        all_positions = [q_start.copy()]
        all_velocities = [np.zeros_like(q_start)]
        all_accelerations = [np.zeros_like(q_start)]
        all_times = [0.0]
        t_offset = 0.0

        for seg in range(n_seg):
            T = durs[seg]
            p0, p1 = wps[seg], wps[seg + 1]
            v0, v1 = vels[seg], vels[seg + 1]

            n_samples = max(self.min_samples, int(np.ceil(T / self.sample_period)) + 1)
            ts = np.linspace(0.0, T, n_samples)

            for k in range(1, n_samples):
                t = ts[k]
                tau = t / T
                tau2 = tau * tau
                tau3 = tau2 * tau

                # Cubic Hermite basis
                h00 = 2 * tau3 - 3 * tau2 + 1
                h10 = tau3 - 2 * tau2 + tau
                h01 = -2 * tau3 + 3 * tau2
                h11 = tau3 - tau2

                pos = h00 * p0 + h10 * T * v0 + h01 * p1 + h11 * T * v1

                # Velocity (d/dt of Hermite)
                dh00 = (6 * tau2 - 6 * tau) / T
                dh10 = (3 * tau2 - 4 * tau + 1) / T
                dh01 = (-6 * tau2 + 6 * tau) / T
                dh11 = (3 * tau2 - 2 * tau) / T
                vel = dh00 * p0 + dh10 * T * v0 + dh01 * p1 + dh11 * T * v1

                # Acceleration (d²/dt² of Hermite)
                ddh00 = (12 * tau - 6) / (T * T)
                ddh10 = (6 * tau - 4) / (T * T)
                ddh01 = (-12 * tau + 6) / (T * T)
                ddh11 = (6 * tau - 2) / (T * T)
                acc = ddh00 * p0 + ddh10 * T * v0 + ddh01 * p1 + ddh11 * T * v1

                all_positions.append(pos)
                all_velocities.append(vel)
                all_accelerations.append(acc)
                all_times.append(t_offset + t)

            t_offset += T

        positions = np.array(all_positions, dtype=float)
        velocities = np.array(all_velocities, dtype=float)
        accelerations = np.array(all_accelerations, dtype=float)
        times = np.array(all_times, dtype=float)

        # Time-scale if limits are violated
        scaled_times, scale = self._limit_time_scaling(times, velocities, accelerations)
        if scale > 1.0:
            times = scaled_times
            velocities /= scale
            accelerations /= (scale * scale)

        return self._trajectory_from_samples(positions, velocities, accelerations, times)

    def plan_cartesian_trajectory(
        self,
        target_position: np.ndarray,
        duration: float,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_cartesian_step: Optional[float] = None,
        ik_tolerance: float = 2e-3,
        postprocess_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[JointTrajectory]:
        if self.ik_solver is None:
            self.node.get_logger().error('Cartesian planning requested before IK solver was initialized')
            return None

        q_start = self.current_joints.copy() if q_init is None else np.array(q_init, dtype=float)
        start_pose = self.ik_solver.forward_kinematics(q_start)
        start_pos = start_pose.translation.copy()
        goal_pos = np.array(target_position, dtype=float)

        path_length = float(np.linalg.norm(goal_pos - start_pos))
        max_step = self.max_cartesian_step if max_cartesian_step is None else float(max_cartesian_step)
        min_samples = max(self.min_samples, int(np.ceil(path_length / max(max_step, 1e-3))) + 1)
        _, times, sigma, _, _ = self._sample_quintic_profile(duration, min_samples=min_samples)

        waypoints = start_pos[None, :] + sigma[:, None] * (goal_pos - start_pos)[None, :]

        joint_samples = np.zeros((len(times), len(self.joint_names)), dtype=float)
        joint_samples[0] = q_start
        q_prev = q_start.copy()

        for idx in range(1, len(times)):
            waypoint = waypoints[idx]
            success, q_sol = self.ik_solver.solve(
                waypoint,
                target_orientation,
                q_prev,
                max_iterations=300,
                tolerance=ik_tolerance,
                damping=1e-4,
                step_size=0.5,
            )
            if not success:
                success, q_sol = self.ik_solver.solve_z_axis_down(
                    waypoint,
                    q_prev,
                    max_iterations=400,
                    tolerance=max(ik_tolerance, 3e-3),
                    damping=1e-4,
                    step_size=0.5,
                )
            if not success:
                self.node.get_logger().warn(
                    f'Cartesian IK failed at sample {idx}/{len(times)-1}: '
                    f'target={np.round(waypoint, 4)}'
                )
                return None

            if postprocess_q is not None:
                q_sol = np.array(postprocess_q(np.array(q_sol, dtype=float)), dtype=float)

            joint_samples[idx] = q_sol
            q_prev = q_sol

        velocities, accelerations = self._numerical_derivatives(joint_samples, times)
        scaled_times, scale = self._limit_time_scaling(times, velocities, accelerations)
        if scale > 1.0:
            times = scaled_times
            velocities, accelerations = self._numerical_derivatives(joint_samples, times)
        return self._trajectory_from_samples(joint_samples, velocities, accelerations, times)

    def plan_cartesian_polyline(
        self,
        waypoints: np.ndarray,
        segment_duration,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_cartesian_step: Optional[float] = None,
        ik_tolerance: float = 2e-3,
        postprocess_q: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> Optional[JointTrajectory]:
        if self.ik_solver is None:
            self.node.get_logger().error('Cartesian polyline planning requested before IK solver was initialized')
            return None

        waypoints = [np.array(p, dtype=float) for p in waypoints]
        if not waypoints:
            return None

        if np.isscalar(segment_duration):
            segment_durations = [float(segment_duration)] * len(waypoints)
        else:
            segment_durations = [float(d) for d in segment_duration]
            if len(segment_durations) != len(waypoints):
                self.node.get_logger().error(
                    'Cartesian polyline planning requires one duration per waypoint segment'
                )
                return None
        segment_durations = [
            max(float(d), self.sample_period * 3.0) for d in segment_durations
        ]

        q_start = self.current_joints.copy() if q_init is None else np.array(q_init, dtype=float)
        start_pose = self.ik_solver.forward_kinematics(q_start)
        current_pos = start_pose.translation.copy()
        q_prev = q_start.copy()
        max_step = self.max_cartesian_step if max_cartesian_step is None else float(max_cartesian_step)

        segment_starts = [current_pos.copy()]
        segment_lengths = []
        for goal_pos in waypoints:
            goal_pos = np.array(goal_pos, dtype=float)
            segment_lengths.append(float(np.linalg.norm(goal_pos - current_pos)))
            current_pos = goal_pos.copy()
            segment_starts.append(current_pos.copy())

        total_length = float(np.sum(segment_lengths))
        if total_length <= 1e-9:
            return self.plan_joint_trajectory(q_start, sum(segment_durations), q_init=q_start)

        total_duration = max(
            float(np.sum(segment_durations)),
            self.sample_period * max(self.min_samples - 1, 3),
        )
        min_samples = max(
            self.min_samples,
            int(np.ceil(total_length / max(max_step, 1e-3))) + 1,
        )
        _, times, sigma, _, _ = self._sample_quintic_profile(
            total_duration,
            min_samples=min_samples,
        )
        arc_samples = sigma * total_length
        cumulative_lengths = np.cumsum(segment_lengths)

        sample_positions = [q_start.copy()]
        sample_times = [0.0]

        for sample_idx in range(1, len(times)):
            arc_pos = float(arc_samples[sample_idx])
            seg_idx = int(np.searchsorted(cumulative_lengths, arc_pos, side='right'))
            seg_idx = min(seg_idx, len(waypoints) - 1)
            seg_start_s = 0.0 if seg_idx == 0 else float(cumulative_lengths[seg_idx - 1])
            seg_length = max(segment_lengths[seg_idx], 1e-9)
            seg_alpha = np.clip((arc_pos - seg_start_s) / seg_length, 0.0, 1.0)
            seg_start = segment_starts[seg_idx]
            seg_goal = waypoints[seg_idx]
            waypoint = seg_start + seg_alpha * (seg_goal - seg_start)

            success, q_sol = self.ik_solver.solve(
                waypoint,
                target_orientation,
                q_prev,
                max_iterations=300,
                tolerance=ik_tolerance,
                damping=1e-4,
                step_size=0.5,
            )
            if not success:
                success, q_sol = self.ik_solver.solve_z_axis_down(
                    waypoint,
                    q_prev,
                    max_iterations=400,
                    tolerance=max(ik_tolerance, 3e-3),
                    damping=1e-4,
                    step_size=0.5,
                )
            if not success:
                self.node.get_logger().warn(
                    f'Cartesian polyline IK failed at segment {seg_idx + 1}/{len(waypoints)} '
                    f'sample {sample_idx}/{len(times)-1}: target={np.round(waypoint, 4)}'
                )
                return None

            if postprocess_q is not None:
                q_sol = np.array(postprocess_q(np.array(q_sol, dtype=float)), dtype=float)

            sample_positions.append(q_sol.copy())
            sample_times.append(float(times[sample_idx]))
            q_prev = q_sol

        positions = np.array(sample_positions, dtype=float)
        times = np.array(sample_times, dtype=float)
        velocities, accelerations = self._numerical_derivatives(positions, times)
        scaled_times, scale = self._limit_time_scaling(times, velocities, accelerations)
        if scale > 1.0:
            times = scaled_times
            velocities, accelerations = self._numerical_derivatives(positions, times)
        return self._trajectory_from_samples(positions, velocities, accelerations, times)

    def execute_trajectory_blocking(self, trajectory: Optional[JointTrajectory]) -> bool:
        if trajectory is None:
            self.node.get_logger().error('Refusing to send null trajectory')
            return False
        try:
            if not self.action_client.server_is_ready():
                self.node.get_logger().error('Arm trajectory action server not ready')
                return False
            if not trajectory.points:
                self.node.get_logger().warn('Refusing to send empty trajectory')
                return False

            goal = FollowJointTrajectory.Goal()
            goal.trajectory = trajectory

            send_future = self.action_client.send_goal_async(goal)
            goal_handle = self._wait_for_future(send_future)
            if goal_handle is None:
                self.node.get_logger().warn('Trajectory send interrupted during shutdown')
                return False
            if not goal_handle.accepted:
                self.node.get_logger().error('Trajectory goal rejected')
                return False

            result_future = goal_handle.get_result_async()
            result = self._wait_for_future(result_future)
            if result is None:
                self.node.get_logger().warn('Trajectory result wait interrupted during shutdown')
                return False
            if result.result is None:
                self.node.get_logger().error('Trajectory execution returned no result payload')
                return False

            error_code = result.result.error_code
            if error_code != FollowJointTrajectory.Result.SUCCESSFUL:
                self.node.get_logger().warn(
                    f'Trajectory execution finished with controller error code {error_code}'
                )
                return False

            self.update_current_joints(np.array(trajectory.points[-1].positions, dtype=float))
            return True
        except Exception as exc:
            self.node.get_logger().error(f'Trajectory execution failed: {exc}')
            return False
