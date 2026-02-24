#!/usr/bin/env python3
"""
gripper_controller.py
Controls the Gen3 Lite 2F gripper via GripperCommand action.

Uses the GripperActionController which commands right_finger_bottom_joint.
The gz_ros2_control hardware interface handles the mimic joints internally,
driving all 4 finger joints from the single commanded joint.
"""
import asyncio
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from control_msgs.action import GripperCommand


class GripperController:
    """
    Controls the Gen3 Lite 2F gripper via GripperCommand action.

    Commands right_finger_bottom_joint only — the gz_ros2_control hardware
    interface handles mimic joints (tip + left finger) internally.
    """

    def __init__(
        self,
        node: Node,
        action_name: str = '/gripper_controller/gripper_cmd',
        open_position: float = 0.0,
        close_position: float = 0.8,
        max_effort: float = 100.0,
        wait_after_command: float = 1.0,
    ):
        """
        Initialize the gripper controller.

        Args:
            node: Parent ROS2 node
            action_name: GripperCommand action topic
            open_position: Finger position for fully open (0.0)
            close_position: Finger position for closed-on-cube (~0.8)
            max_effort: Maximum grip effort
            wait_after_command: Seconds to wait after gripper command
        """
        self.node = node
        self.open_pos = open_position
        self.close_pos = close_position
        self.max_effort = max_effort
        self.wait_time = wait_after_command

        # Current finger position (updated by joint state callback)
        self.current_finger_pos = 0.0

        # Action client
        self.action_client = ActionClient(
            node, GripperCommand, action_name
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

    async def _send_gripper_command(self, position: float, effort: float = -1.0) -> bool:
        """
        Send a GripperCommand and wait for completion.

        Args:
            position: Target finger position
            effort: Maximum effort (-1.0 for default max)

        Returns:
            True if command succeeded
        """
        if not self.action_client.server_is_ready():
            self.node.get_logger().error('Gripper action server not ready!')
            return False

        goal = GripperCommand.Goal()
        goal.command.position = position
        goal.command.max_effort = effort if effort > 0 else self.max_effort

        self.node.get_logger().info(
            f'Sending gripper command: position={position:.2f}, effort={goal.command.max_effort:.1f}'
        )

        send_goal_future = self.action_client.send_goal_async(goal)
        goal_handle = await send_goal_future

        if not goal_handle.accepted:
            self.node.get_logger().error('Gripper goal rejected!')
            return False

        result_future = goal_handle.get_result_async()
        result = await result_future

        # Wait additional time for gripper to stabilize
        await asyncio.sleep(self.wait_time)

        self.node.get_logger().info(
            f'Gripper command done: position={result.result.position:.3f}, '
            f'stalled={result.result.stalled}, reached_goal={result.result.reached_goal}'
        )

        return True

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
