#!/usr/bin/env python3
"""
grasp_verifier.py
Multi-method grasp verification for the Kinova Gen3 Lite.

Uses at least TWO independent methods to verify whether a cube has been
successfully grasped:
  1. Gripper finger gap check (joint position analysis)
  2. Visual verification (cube no longer detected at original position)
"""
import numpy as np
from typing import Optional, Tuple
from geometry_msgs.msg import PoseArray


class GraspVerifier:
    """
    Verifies grasp success using multiple independent methods.

    Must pass at least 2 out of 2 checks to confirm a successful grasp.
    """

    def __init__(
        self,
        finger_min_expected: float = 0.4,
        finger_max_expected: float = 0.75,
        position_tolerance: float = 0.03,
        required_checks: int = 2,
    ):
        """
        Initialize the grasp verifier.

        Args:
            finger_min_expected: Min finger position when cube is held
            finger_max_expected: Max finger position when cube is held
            position_tolerance: Distance tolerance for "cube gone from table" check
            required_checks: Number of checks that must pass
        """
        self.finger_min = finger_min_expected
        self.finger_max = finger_max_expected
        self.pos_tolerance = position_tolerance
        self.required_checks = required_checks

    def check_finger_gap(self, finger_position: float) -> bool:
        """
        Method 1: Check if gripper finger position indicates a cube is held.

        If the fingers stopped at a position within the expected range
        (not fully closed, not fully open), it suggests a cube is between them.

        Args:
            finger_position: Current right_finger_bottom_joint position
                            (0.0=open, 0.85=fully closed)

        Returns:
            True if finger position is consistent with holding a cube
        """
        is_in_range = self.finger_min <= finger_position <= self.finger_max
        return is_in_range

    def check_visual(
        self,
        cube_detections: Optional[PoseArray],
        target_color: str,
        original_position: np.ndarray,
    ) -> bool:
        """
        Method 2: Check if the cube is no longer detected at its original position.

        If the cube has been picked up, it should no longer be visible at its
        original table position.

        Args:
            cube_detections: Latest PoseArray from /perception/cube_detections
            target_color: Color of the target cube ('red', 'green', 'yellow')
            original_position: Original world-frame position of the cube [x, y, z]

        Returns:
            True if cube is NOT detected at original position (likely grasped)
        """
        if cube_detections is None:
            return False  # No detection data, can't verify

        # Color is encoded in the orientation quaternion:
        # red=(1,0,0,0), green=(0,1,0,0), yellow=(0,0,1,0)
        color_map = {
            'red': lambda q: q.x > 0.5,
            'green': lambda q: q.y > 0.5,
            'yellow': lambda q: q.z > 0.5,
        }

        check_fn = color_map.get(target_color)
        if check_fn is None:
            return False

        for pose in cube_detections.poses:
            if check_fn(pose.orientation):
                # Found a detection of this color — check if it's at the original position
                det_pos = np.array([
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ])
                distance = np.linalg.norm(det_pos[:2] - original_position[:2])
                if distance < self.pos_tolerance:
                    # Cube is still at its original position — grasp likely failed
                    return False

        # Cube not detected at original position — likely in the gripper
        return True

    def verify_grasp(
        self,
        finger_position: float,
        cube_detections: Optional[PoseArray],
        target_color: str,
        original_position: np.ndarray,
    ) -> Tuple[bool, int, str]:
        """
        Run all grasp verification checks.

        Args:
            finger_position: Current finger joint position
            cube_detections: Latest perception data
            target_color: Color of the target cube
            original_position: Original position of the target cube

        Returns:
            (success, checks_passed, reason_string)
        """
        checks_passed = 0
        reasons = []

        # Check 1: Finger gap
        finger_ok = self.check_finger_gap(finger_position)
        if finger_ok:
            checks_passed += 1
            reasons.append(f'finger_gap OK (pos={finger_position:.3f})')
        else:
            reasons.append(f'finger_gap FAIL (pos={finger_position:.3f}, '
                          f'expected {self.finger_min:.2f}-{self.finger_max:.2f})')

        # Check 2: Visual verification
        visual_ok = self.check_visual(cube_detections, target_color, original_position)
        if visual_ok:
            checks_passed += 1
            reasons.append('visual OK (cube gone from original pos)')
        else:
            reasons.append('visual FAIL (cube still at original pos or no data)')

        success = checks_passed >= self.required_checks
        reason_str = '; '.join(reasons)
        return success, checks_passed, reason_str


class DropDetector:
    """
    Monitors for cube drops during transit by checking finger positions.
    """

    def __init__(
        self,
        finger_min_threshold: float = 0.3,
        finger_max_threshold: float = 0.85,
    ):
        """
        Initialize the drop detector.

        Args:
            finger_min_threshold: Below this = fingers too open (cube dropped)
            finger_max_threshold: Above this = fingers fully closed (cube slipped)
        """
        self.min_threshold = finger_min_threshold
        self.max_threshold = finger_max_threshold

    def check_for_drop(self, finger_position: float) -> bool:
        """
        Check if the cube has been dropped based on finger position.

        Args:
            finger_position: Current finger joint position

        Returns:
            True if a drop is detected
        """
        if finger_position < self.min_threshold:
            # Fingers too open — cube fell out
            return True
        if finger_position > self.max_threshold:
            # Fingers fully closed — cube slipped through
            return True
        return False
