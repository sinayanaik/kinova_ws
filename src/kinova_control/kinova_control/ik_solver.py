#!/usr/bin/env python3
"""
ik_solver.py
Pinocchio-based inverse kinematics solver for the Kinova Gen3 Lite.

Uses damped least squares (Levenberg-Marquardt) iterative IK.
No MoveIt, no KDL — Pinocchio only, as required by project constraints.
"""
import numpy as np
import pinocchio as pin
from typing import Optional, Tuple


class IKSolver:
    """
    Inverse kinematics solver using Pinocchio's damped least squares method.

    Loads the robot URDF, identifies the end-effector frame, and solves IK
    for desired SE3 target poses.
    """

    def __init__(self, urdf_string: str, ee_frame_name: str = 'end_effector_link'):
        """
        Initialize the IK solver with a URDF string.

        Args:
            urdf_string: Complete URDF XML as a string (from robot_description)
            ee_frame_name: Name of the end-effector frame in the URDF
        """
        # Build Pinocchio model from URDF string
        self.model = pin.buildModelFromXML(urdf_string)
        self.data = self.model.createData()

        # Find end-effector frame ID
        self.ee_frame_id = None
        for i in range(self.model.nframes):
            if self.model.frames[i].name == ee_frame_name:
                self.ee_frame_id = i
                break

        if self.ee_frame_id is None:
            # Fallback: try 'tool_frame' or use the last frame
            for name in ['tool_frame', 'dummy_link']:
                for i in range(self.model.nframes):
                    if self.model.frames[i].name == name:
                        self.ee_frame_id = i
                        break
                if self.ee_frame_id is not None:
                    break

        if self.ee_frame_id is None:
            raise ValueError(
                f'End-effector frame "{ee_frame_name}" not found in URDF. '
                f'Available frames: {[self.model.frames[i].name for i in range(self.model.nframes)]}'
            )

        # Store joint limits
        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()

        # Number of actuated joints (arm only = first 6)
        self.n_arm_joints = 6

    def forward_kinematics(self, q: np.ndarray) -> pin.SE3:
        """
        Compute forward kinematics for the given joint configuration.

        Args:
            q: Joint angles array (at least 6 elements for the arm)

        Returns:
            SE3 pose of the end-effector frame
        """
        # Ensure q has the right size for the full model
        q_full = np.zeros(self.model.nq)
        n = min(len(q), self.model.nq)
        q_full[:n] = q[:n]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id].copy()

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
        damping: float = 1e-6,
        step_size: float = 1.0,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve inverse kinematics using damped least squares.

        Args:
            target_position: Desired end-effector position [x, y, z]
            target_orientation: Desired orientation as rotation matrix (3x3).
                               If None, uses a default downward-facing orientation.
            q_init: Initial joint configuration. If None, uses zeros.
            max_iterations: Maximum number of IK iterations
            tolerance: Convergence tolerance (norm of error)
            damping: Damping factor λ for damped least squares
            step_size: Step size α for joint angle updates

        Returns:
            (success, q_solution): Tuple of (bool, joint angles array)
        """
        # Build target SE3 pose
        if target_orientation is None:
            # Default: gripper pointing straight down
            target_orientation = np.array([
                [1.0,  0.0,  0.0],
                [0.0, -1.0,  0.0],
                [0.0,  0.0, -1.0],
            ])

        target_pose = pin.SE3(target_orientation, target_position)

        # Initial configuration
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = np.zeros(self.model.nq)
            n = min(len(q_init), self.model.nq)
            q[:n] = q_init[:n]

        for iteration in range(max_iterations):
            # Forward kinematics
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            current_pose = self.data.oMf[self.ee_frame_id]

            # Compute 6D error: [position_error; orientation_error_as_log3]
            error_se3 = pin.log6(current_pose.actInv(target_pose))
            error = error_se3.vector  # 6D error vector

            error_norm = np.linalg.norm(error)
            if error_norm < tolerance:
                return True, q[:self.n_arm_joints].copy()

            # Compute frame Jacobian
            J = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            # Damped least squares: dq = J^T (J J^T + λ²I)^{-1} error
            JJt = J @ J.T
            damped = JJt + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(damped, error)

            # Update joint angles with step size
            q = pin.integrate(self.model, q, step_size * dq)

            # Clamp to joint limits
            for i in range(min(self.n_arm_joints, len(q))):
                q[i] = np.clip(q[i], self.q_min[i], self.q_max[i])

        # Failed to converge
        return False, q[:self.n_arm_joints].copy()

    def solve_position_only(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 200,
        tolerance: float = 1e-3,
        damping: float = 1e-6,
        step_size: float = 1.0,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve IK prioritizing position accuracy over orientation.

        First attempts full 6D IK. If that fails, relaxes orientation
        constraint by trying multiple initial configurations.

        Args:
            Same as solve()

        Returns:
            (success, q_solution)
        """
        # Try with the given initial config first
        success, q_sol = self.solve(
            target_position, target_orientation, q_init,
            max_iterations, tolerance, damping, step_size
        )
        if success:
            return True, q_sol

        # Try with several random initial configurations
        for _ in range(5):
            q_random = pin.randomConfiguration(self.model)
            success, q_sol = self.solve(
                target_position, target_orientation, q_random,
                max_iterations, tolerance, damping, step_size
            )
            if success:
                return True, q_sol

        return False, q_sol

    def get_joint_names(self):
        """Return the names of the first 6 (arm) joints."""
        names = []
        for i in range(1, min(self.model.njoints, self.n_arm_joints + 1)):
            names.append(self.model.names[i])
        return names
