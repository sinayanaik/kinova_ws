#!/usr/bin/env python3
"""
gripper_controller.py
Controls the Gen3 Lite 2F gripper via the gripper_trajectory_controller.

Since Gazebo Harmonic doesn't support URDF mimic joints, this controller
commands all 4 finger joints independently to achieve real physical grasping.
The finger tip joints are computed from the bottom joint using the mimic
relationship from the original URDF.
"""
import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from builtin_interfaces.msg import Duration
from typing import Optional


class GripperController:
    """
    Controls the Gen3 Lite 2F gripper for real physical grasping in Gazebo.

    Commands all 4 gripper joints (right/left finger bottom + tip) via the
    gripper_trajectory_controller. Tip joint positions are computed from the
    bottom joint using the kinematic coupling (mimic) relationship.
    """

    # Mimic joint relationships from the official URDF:
    # right_finger_tip_joint: mimic right_finger_bottom_joint * (-0.276) + (-0.1)
    # left_finger_bottom_joint: mimic right_finger_bottom_joint * 1.0 + 0.0
    # left_finger_tip_joint: mimic right_finger_bottom_joint * (-0.276) + (-0.1)

    JOINT_NAMES = [
        'right_finger_bottom_joint',
        'right_finger_tip_joint',
        'left_finger_bottom_joint',
        'left_finger_tip_joint',
    ]

    def __init__(
        self,
        node: Node,
        action_name: str = '/gripper_trajectory_controller/follow_joint_trajectory',
        open_position: float = 0.0,
        close_position: float = 0.8,
        command_duration: float = 1.5,
        wait_after_command: float = 1.0,
    ):
        """
        Initialize the gripper controller.

        Args:
            node: Parent ROS2 node
            action_name: FollowJointTrajectory action topic for gripper
            open_position: Finger position for fully open (0.0)
            close_position: Finger position for closed-on-cube (~0.8)
            command_duration: Duration for gripper movement
            wait_after_command: Seconds to wait after gripper command
        """
        self.node = node
        self.open_pos = open_position
        self.close_pos = close_position
        self.cmd_duration = command_duration
        self.wait_time = wait_after_command

        # Current finger position (updated by joint state callback)
        self.current_finger_pos = 0.0

        # Action client
        self.action_client = ActionClient(
            node, FollowJointTrajectory, action_name
        )
        self.node.get_logger().info(
            f'GripperController: Waiting for action server {action_name}...'
        )

    def wait_for_server(self, timeout_sec: float = 30.0) -> bool:
        """Wait for the gripper action server."""
        return self.action_client.wait_for_server(timeout_sec=timeout_sec)

    def update_finger_position(self, position: float):
        """Update the current finger position (from joint state callback)."""
        self.current_finger_pos = position

    def _compute_joint_positions(self, bottom_joint_pos: float) -> list:
        """
        Compute all 4 gripper joint positions from the commanded bottom joint position.

        Uses the mimic relationships from the official Kinova URDF to ensure
        proper finger linkage motion for realistic grasping.

        Args:
            bottom_joint_pos: Desired position for right_finger_bottom_joint

        Returns:
            List of 4 joint positions [right_bottom, right_tip, left_bottom, left_tip]
        """
        right_bottom = bottom_joint_pos
        # right_finger_tip: mimic * (-0.276) + (-0.1)
        right_tip = -0.276 * bottom_joint_pos + (-0.1)
        # left_finger_bottom: mimic * 1.0
        left_bottom = bottom_joint_pos
        # left_finger_tip: mimic * (-0.276) + (-0.1)
        left_tip = -0.276 * bottom_joint_pos + (-0.1)

        return [right_bottom, right_tip, left_bottom, left_tip]

    def _create_gripper_trajectory(self, target_bottom_pos: float) -> JointTrajectory:
        """
        Create a JointTrajectory message for the gripper.

        Args:
            target_bottom_pos: Target position for the finger bottom joints

        Returns:
            JointTrajectory message for all 4 gripper joints
        """
        positions = self._compute_joint_positions(target_bottom_pos)

        trajectory = JointTrajectory()
        trajectory.joint_names = self.JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = positions
        point.velocities = [0.0] * 4
        point.time_from_start = Duration(
            sec=int(self.cmd_duration),
            nanosec=int((self.cmd_duration % 1) * 1e9)
        )
        trajectory.points.append(point)

        return trajectory

    async def _send_gripper_command(self, target_pos: float) -> bool:
        """
        Send a gripper command and wait for completion.

        Args:
            target_pos: Target bottom joint position

        Returns:
            True if command succeeded
        """
        if not self.action_client.server_is_ready():
            self.node.get_logger().error('Gripper action server not ready!')
            return False

        trajectory = self._create_gripper_trajectory(target_pos)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = trajectory

        self.node.get_logger().info(
            f'Sending gripper command: bottom_joint={target_pos:.2f}'
        )

        send_goal_future = self.action_client.send_goal_async(goal)
        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.node.get_logger().error('Gripper goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        result = await result_future

        # Wait additional time for gripper to stabilize
        await self._async_sleep(self.wait_time)

        return True

    async def _async_sleep(self, duration: float):
        """Async sleep using ROS2 timer."""
        import asyncio
        await asyncio.sleep(duration)

    async def open_gripper(self) -> bool:
        """
        Open the gripper fully.

        Returns:
            True if command succeeded
        """
        self.node.get_logger().info('Opening gripper...')
        return await self._send_gripper_command(self.open_pos)

    async def close_gripper(self) -> bool:
        """
        Close the gripper (for grasping a 0.04m cube).

        Returns:
            True if command succeeded
        """
        self.node.get_logger().info('Closing gripper...')
        return await self._send_gripper_command(self.close_pos)

    def get_finger_position(self) -> float:
        """Return the current finger bottom joint position."""
        return self.current_finger_pos
