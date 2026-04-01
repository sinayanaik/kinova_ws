#!/usr/bin/env python3
"""
ik_solver.py
Pinocchio-based inverse kinematics solver for the Kinova Gen3 Lite.

Uses damped least squares (Levenberg-Marquardt) iterative IK with
multiple restart strategies and orientation-constrained fallback.
Targets tool_frame (fingertip) for direct fingertip positioning.
No MoveIt, no KDL — Pinocchio only.
"""
import numpy as np
import pinocchio as pin
from typing import Optional, Tuple, List


class IKSolver:
    """
    Inverse kinematics solver using Pinocchio's damped least squares method.

    Features:
    - Full 6D (position + orientation) IK
    - 5D IK: position + Z-axis down (ignores rotation around tool Z)
    - Position-only 3D IK (ignores orientation completely)
    - Multiple initial configuration seeds for robustness
    - Joint-limit clamping
    """

    # Pre-defined seed configurations for the Gen3 Lite arm (6 DOF)
    # Tuned for downward-pointing gripper (tool_frame Z ≈ [0,0,-1])
    # Found via brute-force FK scan of the workspace
    SEED_CONFIGS = [
        # Down-pointing configs (j2≈-0.78, j3≈2.3-2.5, j4≈1.1-2.3, j5≈0-0.2)
        np.array([ 0.0, -0.78, 2.50, 2.08,  0.21, 0.0]),   # center, z≈0.72
        np.array([ 0.3, -0.78, 2.50, 2.08,  0.21, 0.0]),   # rotated right
        np.array([-0.3, -0.78, 2.50, 2.08,  0.21, 0.0]),   # rotated left
        np.array([ 0.6, -0.78, 2.50, 2.08,  0.21, 0.0]),   # more right
        np.array([-0.6, -0.78, 2.50, 2.08,  0.21, 0.0]),   # more left
        np.array([ 0.0, -0.78, 2.29, 2.29,  0.21, 0.0]),   # variant
        np.array([ 0.0, -0.78, 2.50, 1.13,  0.00, 0.0]),   # different j4
        np.array([-0.4, -0.78, 2.50, 1.13,  0.00, 0.0]),   # moved left
        # Mirror configs (j3<0, j4<0)
        np.array([ 0.0,  0.78,-2.50,-2.08, -0.21, 0.0]),   # mirror center
        np.array([ 0.3,  0.78,-2.50,-2.08, -0.21, 0.0]),   # mirror right
        np.array([-0.3,  0.78,-2.50,-2.08, -0.21, 0.0]),   # mirror left
        # Higher reach (z≈0.80-0.90, for pre-grasp / place)
        np.array([ 0.0, -0.55, 2.00, 1.64,  0.55, 0.0]),   # z≈0.80
        np.array([ 0.0, -0.18, 2.00, 1.64,  0.91, 0.0]),   # z≈0.90
        # General configs
        np.array([ 0.62,-0.82, 0.75, 1.57, 1.57, 1.64]),   # observe pose, tool-down
        np.array([ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),         # home/zero
    ]

    def __init__(self, urdf_string: str, ee_frame_name: str = 'tool_frame'):
        """
        Initialize the IK solver with a URDF string.

        Args:
            urdf_string: Complete URDF XML as a string (from robot_description)
            ee_frame_name: Name of the end-effector frame in the URDF
        """
        self.model = pin.buildModelFromXML(urdf_string)
        self.data = self.model.createData()

        # Find end-effector frame ID
        self.ee_frame_id = None
        for i in range(self.model.nframes):
            if self.model.frames[i].name == ee_frame_name:
                self.ee_frame_id = i
                break

        if self.ee_frame_id is None:
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

        self.q_min = self.model.lowerPositionLimit.copy()
        self.q_max = self.model.upperPositionLimit.copy()
        self.n_arm_joints = 6

    def forward_kinematics(self, q: np.ndarray) -> pin.SE3:
        """Compute forward kinematics for given joint configuration."""
        q_full = np.zeros(self.model.nq)
        n = min(len(q), self.model.nq)
        q_full[:n] = q[:n]

        pin.forwardKinematics(self.model, self.data, q_full)
        pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
        return self.data.oMf[self.ee_frame_id].copy()

    def _make_q_full(self, q_arm: np.ndarray) -> np.ndarray:
        """Expand arm-only joint angles to full model configuration."""
        q_full = np.zeros(self.model.nq)
        n = min(len(q_arm), self.model.nq)
        q_full[:n] = q_arm[:n]
        return q_full

    def _clamp_joints(self, q: np.ndarray) -> np.ndarray:
        """Clamp joint angles to joint limits."""
        for i in range(min(self.n_arm_joints, len(q))):
            if self.q_min[i] < self.q_max[i]:
                q[i] = np.clip(q[i], self.q_min[i], self.q_max[i])
        return q

    def clamp_arm_joints(self, q_arm: np.ndarray) -> np.ndarray:
        """Clamp an arm-only joint vector to the model joint limits."""
        q_arm = np.array(q_arm, dtype=float).copy()
        return self._clamp_joints(q_arm)

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 300,
        tolerance: float = 1e-3,
        damping: float = 1e-4,
        step_size: float = 0.5,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve full 6D inverse kinematics using damped least squares.

        Args:
            target_position: Desired EE position [x, y, z] in world frame
            target_orientation: Desired orientation as 3x3 rotation matrix
            q_init: Initial joint configuration
            max_iterations: Maximum IK iterations
            tolerance: Convergence tolerance (norm of 6D error)
            damping: Damping factor lambda
            step_size: Step size alpha for updates

        Returns:
            (success, q_solution)
        """
        if target_orientation is None:
            # Default: tool_frame pointing straight down
            # tool_frame has Rz(π/2) offset from end_effector_link
            target_orientation = np.array([
                [ 0.0, -1.0,  0.0],
                [-1.0,  0.0,  0.0],
                [ 0.0,  0.0, -1.0],
            ])

        target_pose = pin.SE3(target_orientation, target_position)

        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = self._make_q_full(q_init)

        for iteration in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            current_pose = self.data.oMf[self.ee_frame_id]

            error_se3 = pin.log6(current_pose.actInv(target_pose))
            error = error_se3.vector
            error_norm = np.linalg.norm(error)

            if error_norm < tolerance:
                return True, q[:self.n_arm_joints].copy()

            # Adaptive damping: increase for large errors
            adaptive_damping = damping * max(1.0, error_norm)

            J = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            JJt = J @ J.T
            damped = JJt + adaptive_damping * np.eye(6)
            dq = J.T @ np.linalg.solve(damped, error)

            q = pin.integrate(self.model, q, step_size * dq)
            q = self._clamp_joints(q)

        return False, q[:self.n_arm_joints].copy()

    def solve_position_only(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 300,
        tolerance: float = 2e-3,
        damping: float = 1e-4,
        step_size: float = 0.5,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve IK for position only (3D), ignoring orientation.

        Uses only the top 3 rows (translation) of the Jacobian.
        Much more likely to converge for constrained poses.

        Returns:
            (success, q_solution)
        """
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = self._make_q_full(q_init)

        target_pos = np.array(target_position)

        for iteration in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            current_pos = self.data.oMf[self.ee_frame_id].translation

            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)

            if error_norm < tolerance:
                return True, q[:self.n_arm_joints].copy()

            adaptive_damping = damping * max(1.0, error_norm * 5.0)

            J_full = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J_pos = J_full[:3, :]  # 3xN translation rows only

            JJt = J_pos @ J_pos.T  # 3x3
            damped = JJt + adaptive_damping * np.eye(3)
            dq = J_pos.T @ np.linalg.solve(damped, pos_error)

            q = pin.integrate(self.model, q, step_size * dq)
            q = self._clamp_joints(q)

        return False, q[:self.n_arm_joints].copy()

    def solve_z_axis_down(
        self,
        target_position: np.ndarray,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 300,
        tolerance: float = 2e-3,
        damping: float = 1e-4,
        step_size: float = 0.5,
    ) -> Tuple[bool, np.ndarray]:
        """
        Solve 5D IK: position + tool Z-axis pointing down.

        Ignores rotation around the tool's Z-axis (wrist roll is free).
        This is easier to solve than full 6D and ensures the gripper
        is vertical, which is the main requirement for grasping.

        Returns:
            (success, q_solution)
        """
        if q_init is None:
            q = pin.neutral(self.model)
        else:
            q = self._make_q_full(q_init)

        target_pos = np.array(target_position)
        desired_z = np.array([0.0, 0.0, -1.0])  # tool Z pointing down

        for iteration in range(max_iterations):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacement(self.model, self.data, self.ee_frame_id)
            current_pose = self.data.oMf[self.ee_frame_id]

            # Position error (3D)
            pos_error = target_pos - current_pose.translation
            pos_norm = np.linalg.norm(pos_error)

            # Z-axis error: cross(current_z, desired_z)
            # This gives a rotation vector that would align the axes
            current_z = current_pose.rotation[:, 2]
            z_error = np.cross(current_z, desired_z)
            z_norm = np.linalg.norm(z_error)

            # Combined error (6D — 3 position + 3 z-axis cross product)
            error = np.concatenate([pos_error, z_error])
            error_norm = np.linalg.norm(error)

            if pos_norm < tolerance and z_norm < 0.05:  # ~3° Z-axis tolerance
                return True, q[:self.n_arm_joints].copy()

            adaptive_damping = damping * max(1.0, error_norm * 2.0)

            J_full = pin.computeFrameJacobian(
                self.model, self.data, q,
                self.ee_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )

            JJt = J_full @ J_full.T
            damped = JJt + adaptive_damping * np.eye(6)
            dq = J_full.T @ np.linalg.solve(damped, error)

            q = pin.integrate(self.model, q, step_size * dq)
            q = self._clamp_joints(q)

        return False, q[:self.n_arm_joints].copy()

    def solve_robust(
        self,
        target_position: np.ndarray,
        target_orientation: Optional[np.ndarray] = None,
        q_init: Optional[np.ndarray] = None,
        max_iterations: int = 300,
        tolerance: float = 1e-3,
    ) -> Tuple[bool, np.ndarray]:
        """
        Robust IK solver that tries multiple strategies and seeds.

        Strategy order:
        1. Full 6D IK with given q_init
        2. Full 6D IK with each seed config
        3. Z-axis-down 5D IK with given q_init  (ensures vertical gripper)
        4. Z-axis-down 5D IK with each seed config
        5. Position-only IK with seeds (last resort)

        Returns:
            (success, q_solution)
        """
        q_sol = np.zeros(self.n_arm_joints)

        # Strategy 1: Full 6D with given q_init
        if q_init is not None:
            success, q_sol = self.solve(
                target_position, target_orientation, q_init,
                max_iterations, tolerance, damping=1e-4, step_size=0.5
            )
            if success:
                return True, q_sol

        # Strategy 2: Full 6D with seed configs
        for seed in self.SEED_CONFIGS:
            success, q_sol = self.solve(
                target_position, target_orientation, seed,
                max_iterations, tolerance, damping=1e-4, step_size=0.5
            )
            if success:
                return True, q_sol

        # Strategy 3: Z-axis-down with given q_init
        if q_init is not None:
            success, q_sol = self.solve_z_axis_down(
                target_position, q_init,
                max_iterations, tolerance * 2, damping=1e-4, step_size=0.5
            )
            if success:
                return True, q_sol

        # Strategy 4: Z-axis-down with seed configs
        for seed in self.SEED_CONFIGS:
            success, q_sol = self.solve_z_axis_down(
                target_position, seed,
                max_iterations, tolerance * 2, damping=1e-4, step_size=0.5
            )
            if success:
                return True, q_sol

        # Strategy 5: Position-only with seeds (last resort, no orientation)
        for seed in self.SEED_CONFIGS:
            success, q_sol = self.solve_position_only(
                target_position, target_orientation, seed,
                max_iterations, tolerance * 2, damping=1e-4, step_size=0.5
            )
            if success:
                return True, q_sol

        # Strategy 6: Random restarts (position-only)
        for _ in range(10):
            q_random = pin.randomConfiguration(self.model)
            success, q_sol = self.solve_position_only(
                target_position, target_orientation, q_random,
                max_iterations, tolerance * 3, damping=1e-3, step_size=0.3
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
