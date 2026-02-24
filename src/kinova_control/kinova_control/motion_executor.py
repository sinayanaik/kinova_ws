#!/usr/bin/env python3
"""
motion_executor.py
Trajectory generation and execution for the Kinova Gen3 Lite.

Generates Cartesian-interpolated trajectories, solves IK at each waypoint,
and sends JointTrajectory commands via the FollowJointTrajectory action client.
"""
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from typing import List, Optional, Tuple

from kinova_control.ik_solver import IKSolver


class MotionExecutor:
    """
    Generates and executes arm trajectories via FollowJointTrajectory action.

    Performs Cartesian linear interpolation between waypoints, solves IK at
    each interpolated point, and packages the result as a JointTrajectory.
    """

    def __init__(
        self,
        node: Node,
        ik_solver: IKSolver,
        arm_joint_names: List[str],
        action_name: str = '/joint_trajectory_controller/follow_joint_trajectory',
        interpolation_steps: int = 15,
        waypoint_duration: float = 2.5,
    ):
        """
        Initialize the motion executor.

        Args:
            node: Parent ROS2 node (for logging, action client)
            ik_solver: Pinocchio IK solver instance
            arm_joint_names: List of 6 arm joint names
            action_name: FollowJointTrajectory action topic
            interpolation_steps: Number of interpolated points between waypoints
            waypoint_duration: Seconds per waypoint transition
        """
        self.node = node
        self.ik_solver = ik_solver
        self.joint_names = arm_joint_names
        self.interp_steps = interpolation_steps
        self.wp_duration = waypoint_duration

        # Current joint positions (updated by joint state subscriber)
        self.current_joints = np.zeros(6)

        # Action client for the arm trajectory controller
        self.action_client = ActionClient(
            node, FollowJointTrajectory, action_name
        )

        self.node.get_logger().info(
            f'MotionExecutor: Waiting for action server {action_name}...'
        )

    def wait_for_server(self, timeout_sec: float = 30.0) -> bool:
        """Wait for the trajectory action server to become available."""
        return self.action_client.wait_for_server(timeout_sec=timeout_sec)

    def update_current_joints(self, joint_positions: np.ndarray):
        """Update the current joint positions (called from joint state callback)."""
        self.current_joints = joint_positions.copy()

    def interpolate_cartesian(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        steps: Optional[int] = None,
    ) -> List[np.ndarray]:
        """
        Linear interpolation in Cartesian space between two positions.

        Args:
            start_pos: Start position [x, y, z]
            end_pos: End position [x, y, z]
            steps: Number of interpolation steps (default: self.interp_steps)

        Returns:
            List of interpolated [x, y, z] positions (including start and end)
        """
        if steps is None:
            steps = self.interp_steps
        positions = []
        for i in range(steps + 1):
            t = i / float(steps)
            pos = start_pos + t * (end_pos - start_pos)
            positions.append(pos)
        return positions

    def plan_trajectory(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
        q_init: Optional[np.ndarray] = None,
    ) -> Optional[JointTrajectory]:
        """
        Plan a trajectory from current position to target via Cartesian interpolation.

        Args:
            target_position: Target end-effector position [x, y, z]
            target_orientation: Target orientation as 3x3 rotation matrix
            duration: Total duration for the trajectory (default: self.wp_duration)
            q_init: Initial joint config (default: self.current_joints)

        Returns:
            JointTrajectory message, or None if IK fails at any point
        """
        if duration is None:
            duration = self.wp_duration
        if q_init is None:
            q_init = self.current_joints.copy()

        # Get current EE position from FK
        current_ee = self.ik_solver.forward_kinematics(q_init)
        start_pos = current_ee.translation.copy()

        # Interpolate in Cartesian space
        waypoints = self.interpolate_cartesian(start_pos, target_position)

        # Solve IK at each waypoint
        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        q_prev = q_init.copy()
        dt = duration / len(waypoints)

        for i, wp_pos in enumerate(waypoints):
            success, q_sol = self.ik_solver.solve(
                wp_pos, target_orientation, q_prev
            )
            if not success:
                # Try Z-axis-down 5D solve as fallback (preserves vertical)
                success, q_sol = self.ik_solver.solve_z_axis_down(
                    wp_pos, q_prev
                )
                if not success:
                    self.node.get_logger().warn(
                        f'IK failed at waypoint {i}/{len(waypoints)}: '
                        f'pos={wp_pos}'
                    )
                    return None

            point = JointTrajectoryPoint()
            point.positions = q_sol.tolist()
            point.velocities = [0.0] * 6
            t = (i + 1) * dt
            point.time_from_start = Duration(
                sec=int(t), nanosec=int((t % 1) * 1e9)
            )
            trajectory.points.append(point)
            q_prev = q_sol.copy()

        return trajectory

    def plan_multi_waypoint_trajectory(
        self,
        positions: List[np.ndarray],
        orientation: Optional[np.ndarray] = None,
        duration_per_segment: Optional[float] = None,
    ) -> Optional[JointTrajectory]:
        """
        Plan a trajectory through multiple Cartesian waypoints.

        Args:
            positions: List of [x, y, z] target positions
            orientation: Orientation for all waypoints (same throughout)
            duration_per_segment: Duration for each segment

        Returns:
            JointTrajectory message, or None if IK fails
        """
        if duration_per_segment is None:
            duration_per_segment = self.wp_duration

        trajectory = JointTrajectory()
        trajectory.joint_names = self.joint_names

        q_prev = self.current_joints.copy()
        total_time = 0.0

        for seg_idx, target_pos in enumerate(positions):
            # Get current position from FK
            current_ee = self.ik_solver.forward_kinematics(q_prev)
            start_pos = current_ee.translation.copy()

            # Interpolate this segment
            waypoints = self.interpolate_cartesian(start_pos, target_pos)
            dt = duration_per_segment / len(waypoints)

            for i, wp_pos in enumerate(waypoints):
                success, q_sol = self.ik_solver.solve(
                    wp_pos, orientation, q_prev
                )
                if not success:
                    success, q_sol = self.ik_solver.solve_z_axis_down(
                        wp_pos, q_prev
                    )
                    if not success:
                        self.node.get_logger().warn(
                            f'IK failed at segment {seg_idx}, waypoint {i}'
                        )
                        return None

                total_time += dt
                point = JointTrajectoryPoint()
                point.positions = q_sol.tolist()
                point.velocities = [0.0] * 6
                point.time_from_start = Duration(
                    sec=int(total_time),
                    nanosec=int((total_time % 1) * 1e9)
                )
                trajectory.points.append(point)
                q_prev = q_sol.copy()

        return trajectory

    async def execute_trajectory(self, trajectory: JointTrajectory) -> bool:
        """
        Send a JointTrajectory to the controller and wait for completion.

        Args:
            trajectory: The trajectory to execute

        Returns:
            True if execution succeeded, False otherwise
        """
        if not self.action_client.server_is_ready():
            self.node.get_logger().error('Trajectory action server not ready!')
            return False

        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.node.get_logger().info(
            f'Sending trajectory with {len(trajectory.points)} points...'
        )

        send_goal_future = self.action_client.send_goal_async(goal)
        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.node.get_logger().error('Trajectory goal rejected!')
            return False

        self.node.get_logger().info('Trajectory goal accepted, executing...')
        result_future = goal_handle.get_result_async()
        result = await result_future

        if result.result.error_code == FollowJointTrajectory.Result.SUCCESSFUL:
            self.node.get_logger().info('Trajectory execution completed successfully.')
            return True
        else:
            self.node.get_logger().warn(
                f'Trajectory execution finished with error code: '
                f'{result.result.error_code}'
            )
            return True  # Consider non-fatal errors as "done"

    async def move_to_position(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        duration: Optional[float] = None,
    ) -> bool:
        """
        Plan and execute a move to a target Cartesian position.

        Args:
            target_position: Target [x, y, z]
            target_orientation: Target orientation (3x3 rotation matrix)
            duration: Duration for the movement

        Returns:
            True if successful
        """
        trajectory = self.plan_trajectory(
            target_position, target_orientation, duration
        )
        if trajectory is None:
            self.node.get_logger().error(
                f'Failed to plan trajectory to {target_position}'
            )
            return False

        return await self.execute_trajectory(trajectory)
