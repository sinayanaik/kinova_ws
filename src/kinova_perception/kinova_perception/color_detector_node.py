#!/usr/bin/env python3
"""
color_detector_node.py
ROS2 node for color-based cube detection and 3D pose estimation.

Uses a SINGLE overhead camera (offset to avoid arm obstruction) to detect
red/green/yellow cubes via HSV segmentation. Back-projects 2D pixel centroids
to 3D world coordinates using the known table height (Z is deterministic).

No side camera, no multi-camera fusion — simple and robust.
"""
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray, Pose, Point, Quaternion
from std_msgs.msg import Header
from cv_bridge import CvBridge
import tf2_ros


class ColorDetectorNode(Node):
    """Detects colored cubes from a single overhead camera image."""

    def __init__(self):
        super().__init__('color_detector')
        self.get_logger().info('Initializing single-camera color detector...')

        self.bridge = CvBridge()

        # TF2 for end-effector tracking
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._declare_parameters()
        self._load_parameters()

        # ---- Single overhead camera subscriber ----
        self.overhead_sub = self.create_subscription(
            Image, '/camera_overhead/image_raw',
            self._overhead_callback, 10
        )

        # ---- Publishers ----
        self.overhead_pub = self.create_publisher(
            Image, '/camera_overhead/processed', 10
        )
        self.detections_pub = self.create_publisher(
            PoseArray, '/perception/cube_detections', 10
        )
        self.ee_pub = self.create_publisher(
            Pose, '/perception/ee_position', 10
        )

        # Detection history for stability filtering
        self.detection_history = {}  # color -> list of recent (x, y, z)

        # Periodic EE pose publisher
        self.create_timer(0.1, self._publish_ee_pose)

        self.get_logger().info('Single-camera color detector ready.')

    # ======================== PARAMETERS ========================

    def _declare_parameters(self):
        """Declare all ROS2 parameters with defaults."""
        # HSV ranges - Red (wraps around 0/180)
        self.declare_parameter('hsv_ranges.red.h_min_1', 0)
        self.declare_parameter('hsv_ranges.red.h_max_1', 10)
        self.declare_parameter('hsv_ranges.red.h_min_2', 170)
        self.declare_parameter('hsv_ranges.red.h_max_2', 180)
        self.declare_parameter('hsv_ranges.red.s_min', 100)
        self.declare_parameter('hsv_ranges.red.s_max', 255)
        self.declare_parameter('hsv_ranges.red.v_min', 100)
        self.declare_parameter('hsv_ranges.red.v_max', 255)

        # HSV ranges - Green
        self.declare_parameter('hsv_ranges.green.h_min', 35)
        self.declare_parameter('hsv_ranges.green.h_max', 85)
        self.declare_parameter('hsv_ranges.green.s_min', 100)
        self.declare_parameter('hsv_ranges.green.s_max', 255)
        self.declare_parameter('hsv_ranges.green.v_min', 100)
        self.declare_parameter('hsv_ranges.green.v_max', 255)

        # HSV ranges - Yellow
        self.declare_parameter('hsv_ranges.yellow.h_min', 20)
        self.declare_parameter('hsv_ranges.yellow.h_max', 40)
        self.declare_parameter('hsv_ranges.yellow.s_min', 100)
        self.declare_parameter('hsv_ranges.yellow.s_max', 255)
        self.declare_parameter('hsv_ranges.yellow.v_min', 150)
        self.declare_parameter('hsv_ranges.yellow.v_max', 255)

        # Contour filtering
        self.declare_parameter('contour_area_min', 200)
        self.declare_parameter('contour_area_max', 5000)

        # Camera intrinsics (must match Gazebo sensor)
        self.declare_parameter('camera_fx', 554.25)
        self.declare_parameter('camera_fy', 554.25)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # Overhead camera world position (must match SDF)
        self.declare_parameter('overhead_camera.x', 0.18)
        self.declare_parameter('overhead_camera.y', 0.00)
        self.declare_parameter('overhead_camera.z', 1.5)

        # Table / cube geometry
        self.declare_parameter('table_height', 0.75)
        self.declare_parameter('cube_half_height', 0.02)

        # Stability filter
        self.declare_parameter('position_stability_readings', 3)
        self.declare_parameter('position_stability_tolerance', 0.015)

    def _load_parameters(self):
        """Load parameter values into instance variables."""
        # HSV ranges
        self.hsv_ranges = {}

        self.hsv_ranges['red'] = {
            'ranges': [
                (
                    np.array([self.get_parameter('hsv_ranges.red.h_min_1').value,
                              self.get_parameter('hsv_ranges.red.s_min').value,
                              self.get_parameter('hsv_ranges.red.v_min').value]),
                    np.array([self.get_parameter('hsv_ranges.red.h_max_1').value,
                              self.get_parameter('hsv_ranges.red.s_max').value,
                              self.get_parameter('hsv_ranges.red.v_max').value]),
                ),
                (
                    np.array([self.get_parameter('hsv_ranges.red.h_min_2').value,
                              self.get_parameter('hsv_ranges.red.s_min').value,
                              self.get_parameter('hsv_ranges.red.v_min').value]),
                    np.array([self.get_parameter('hsv_ranges.red.h_max_2').value,
                              self.get_parameter('hsv_ranges.red.s_max').value,
                              self.get_parameter('hsv_ranges.red.v_max').value]),
                ),
            ],
            'bgr': (0, 0, 255),
        }

        self.hsv_ranges['green'] = {
            'ranges': [
                (
                    np.array([self.get_parameter('hsv_ranges.green.h_min').value,
                              self.get_parameter('hsv_ranges.green.s_min').value,
                              self.get_parameter('hsv_ranges.green.v_min').value]),
                    np.array([self.get_parameter('hsv_ranges.green.h_max').value,
                              self.get_parameter('hsv_ranges.green.s_max').value,
                              self.get_parameter('hsv_ranges.green.v_max').value]),
                ),
            ],
            'bgr': (0, 255, 0),
        }

        self.hsv_ranges['yellow'] = {
            'ranges': [
                (
                    np.array([self.get_parameter('hsv_ranges.yellow.h_min').value,
                              self.get_parameter('hsv_ranges.yellow.s_min').value,
                              self.get_parameter('hsv_ranges.yellow.v_min').value]),
                    np.array([self.get_parameter('hsv_ranges.yellow.h_max').value,
                              self.get_parameter('hsv_ranges.yellow.s_max').value,
                              self.get_parameter('hsv_ranges.yellow.v_max').value]),
                ),
            ],
            'bgr': (0, 255, 255),
        }

        self.area_min = self.get_parameter('contour_area_min').value
        self.area_max = self.get_parameter('contour_area_max').value

        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value

        self.overhead_cam_pos = np.array([
            self.get_parameter('overhead_camera.x').value,
            self.get_parameter('overhead_camera.y').value,
            self.get_parameter('overhead_camera.z').value,
        ])

        self.table_height = self.get_parameter('table_height').value
        self.cube_half = self.get_parameter('cube_half_height').value

        self.stability_readings = self.get_parameter('position_stability_readings').value
        self.stability_tol = self.get_parameter('position_stability_tolerance').value

    # ======================== COLOR DETECTION ========================

    def _detect_colors(self, bgr_image):
        """
        Run HSV color segmentation on a BGR image.

        Returns dict: color_name -> list of (cx, cy, area) centroid tuples.
        Each color can have multiple detections (multiple cubes of same color).
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        detections = {}

        kernel = np.ones((5, 5), np.uint8)

        for color_name, params in self.hsv_ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in params['ranges']:
                mask |= cv2.inRange(hsv, lower, upper)

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            color_dets = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if self.area_min <= area <= self.area_max:
                    M = cv2.moments(contour)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        color_dets.append((cx, cy, area))

            if color_dets:
                # Sort by area descending — return ALL valid detections
                color_dets.sort(key=lambda d: d[2], reverse=True)
                detections[color_name] = color_dets

        return detections

    def _annotate_image(self, bgr_image, detections, coord_texts=None):
        """Draw crosshairs and labels on the image for each detection."""
        annotated = bgr_image.copy()

        for color_name, dets in detections.items():
            bgr_color = self.hsv_ranges[color_name]['bgr']
            for i, (cx, cy, area) in enumerate(dets):
                cv2.circle(annotated, (cx, cy), 8, bgr_color, 2)
                cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), bgr_color, 2)
                cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), bgr_color, 2)

                label = f'{color_name.upper()}'
                if len(dets) > 1:
                    label += f'_{i+1}'
                if coord_texts and (color_name, i) in coord_texts:
                    label += f' {coord_texts[(color_name, i)]}'
                cv2.putText(annotated, label, (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr_color, 2)

        return annotated

    # ======================== BACK-PROJECTION ========================

    def _overhead_back_project(self, cx_px, cy_px):
        """
        Back-project a pixel from the overhead camera to world (X, Y, Z).

        The overhead camera points straight down with Ry(pi/2).
        Z is known (table_height + cube_half_height), so we solve for X, Y.

        Camera frame mapping for Ry(pi/2):
          pixel +u (right) -> world -Y
          pixel +v (down)  -> world -X

        Therefore:
          world_x = cam_x - ny * dist
          world_y = cam_y - nx * dist
          world_z = table_height + cube_half_height  (constant)
        """
        z_target = self.table_height + self.cube_half
        dist = self.overhead_cam_pos[2] - z_target

        nx = (cx_px - self.cx) / self.fx
        ny = (cy_px - self.cy) / self.fy

        world_x = self.overhead_cam_pos[0] - ny * dist
        world_y = self.overhead_cam_pos[1] - nx * dist
        world_z = z_target

        return world_x, world_y, world_z

    # ======================== CALLBACKS ========================

    def _overhead_callback(self, msg):
        """Process overhead camera image: detect cubes and publish 3D poses."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        detections = self._detect_colors(bgr)

        # Build PoseArray from all detections
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'world'

        color_quat_map = {
            'red':    Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
            'green':  Quaternion(x=0.0, y=1.0, z=0.0, w=0.0),
            'yellow': Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
        }

        coord_texts = {}

        for color_name, dets in detections.items():
            for i, (cx, cy, area) in enumerate(dets):
                wx, wy, wz = self._overhead_back_project(cx, cy)

                # Spatial filter: only accept cubes in the pick zone
                # (x ∈ [0.10, 0.25], y ∈ [-0.25, 0.25]).
                # Containers are outside this zone and rejected.
                if not (0.10 < wx < 0.25 and -0.25 < wy < 0.25):
                    self.get_logger().debug(
                        f'Filtered {color_name} outside cube zone: '
                        f'({wx:.3f}, {wy:.3f})'
                    )
                    continue

                pose = Pose()
                pose.position = Point(x=float(wx), y=float(wy), z=float(wz))
                pose.orientation = color_quat_map[color_name]
                pose_array.poses.append(pose)

                coord_texts[(color_name, i)] = f'({wx:.2f},{wy:.2f})'

                # Stability tracking
                key = f'{color_name}_{i}'
                if key not in self.detection_history:
                    self.detection_history[key] = []
                self.detection_history[key].append((wx, wy, wz))
                if len(self.detection_history[key]) > self.stability_readings * 3:
                    self.detection_history[key] = \
                        self.detection_history[key][-self.stability_readings * 3:]

        # Publish detections
        if pose_array.poses:
            self.detections_pub.publish(pose_array)

        # Publish annotated image
        annotated = self._annotate_image(bgr, detections, coord_texts)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.overhead_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'CV Bridge publish error: {e}')

    def _publish_ee_pose(self):
        """Look up end-effector pose from TF and publish it."""
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_link', 'end_effector_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            self.ee_pub.publish(pose)
        except Exception:
            pass

    def is_position_stable(self, color_name, idx=0):
        """
        Check if a cube's detected position is stable (consistent readings).

        Returns (stable: bool, position: tuple or None)
        """
        key = f'{color_name}_{idx}'
        if key not in self.detection_history:
            return False, None

        history = self.detection_history[key]
        if len(history) < self.stability_readings:
            return False, None

        recent = history[-self.stability_readings:]
        positions = np.array(recent)
        spread = np.max(positions, axis=0) - np.min(positions, axis=0)

        if np.all(spread < self.stability_tol):
            avg_pos = np.mean(positions, axis=0)
            return True, tuple(avg_pos)

        return False, None


def main(args=None):
    rclpy.init(args=args)
    node = ColorDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()
