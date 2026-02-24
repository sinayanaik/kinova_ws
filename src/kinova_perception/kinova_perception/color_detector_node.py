#!/usr/bin/env python3
"""
color_detector_node.py
ROS2 node for color-based cube detection and 3D pose estimation.

Subscribes to overhead and side camera image topics, performs HSV color
segmentation to identify red/green/yellow cubes, back-projects 2D centroids
to 3D world coordinates, and publishes annotated images + PoseArray detections.

All perception comes from the two cameras — no ground-truth Gazebo data used.
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
    """Detects colored cubes from camera images and estimates 3D world poses."""

    def __init__(self):
        super().__init__('color_detector')
        self.get_logger().info('Initializing color detector node...')

        # ---- CV Bridge ----
        self.bridge = CvBridge()

        # ---- TF2 Buffer for end-effector tracking ----
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---- Declare and load parameters ----
        self._declare_parameters()
        self._load_parameters()

        # ---- Subscribers ----
        self.overhead_sub = self.create_subscription(
            Image, '/camera_overhead/image_raw',
            self._overhead_callback, 10
        )
        self.side_sub = self.create_subscription(
            Image, '/camera_side/image_raw',
            self._side_callback, 10
        )

        # ---- Publishers ----
        self.overhead_pub = self.create_publisher(Image, '/camera_overhead/processed', 10)
        self.side_pub = self.create_publisher(Image, '/camera_side/processed', 10)
        self.detections_pub = self.create_publisher(PoseArray, '/perception/cube_detections', 10)
        self.ee_pub = self.create_publisher(Pose, '/perception/ee_position', 10)

        # ---- State tracking ----
        # Store detections from each camera for fusion
        self.overhead_detections = {}  # color -> (x, y) in world frame
        self.side_detections = {}      # color -> z in world frame
        self.detection_history = {}    # color -> list of recent (x,y,z) readings

        # ---- Periodic EE pose publisher ----
        self.create_timer(0.1, self._publish_ee_pose)

        # ---- Periodic fused detection publisher ----
        self.create_timer(0.1, self._publish_fused_detections)

        self.get_logger().info('Color detector node initialized successfully.')

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

        # Camera intrinsics
        self.declare_parameter('camera_fx', 554.25)
        self.declare_parameter('camera_fy', 554.25)
        self.declare_parameter('camera_cx', 320.0)
        self.declare_parameter('camera_cy', 240.0)
        self.declare_parameter('image_width', 640)
        self.declare_parameter('image_height', 480)

        # Camera extrinsics
        self.declare_parameter('overhead_camera.x', 0.35)
        self.declare_parameter('overhead_camera.y', 0.0)
        self.declare_parameter('overhead_camera.z', 2.25)
        self.declare_parameter('side_camera.x', 0.35)
        self.declare_parameter('side_camera.y', -1.0)
        self.declare_parameter('side_camera.z', 1.25)

        # Table/cube geometry
        self.declare_parameter('table_height', 0.75)
        self.declare_parameter('cube_half_height', 0.02)

        # Detection tuning
        self.declare_parameter('position_stability_readings', 3)
        self.declare_parameter('position_stability_tolerance', 0.01)

    def _load_parameters(self):
        """Load parameter values into instance variables."""
        # Build HSV ranges dictionary
        self.hsv_ranges = {}

        # Red — two ranges because hue wraps around
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
            'bgr': (0, 0, 255),  # for annotation drawing
        }

        # Green
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

        # Yellow
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

        # Contour thresholds
        self.area_min = self.get_parameter('contour_area_min').value
        self.area_max = self.get_parameter('contour_area_max').value

        # Camera intrinsics
        self.fx = self.get_parameter('camera_fx').value
        self.fy = self.get_parameter('camera_fy').value
        self.cx = self.get_parameter('camera_cx').value
        self.cy = self.get_parameter('camera_cy').value

        # Camera extrinsics
        self.overhead_cam_pos = np.array([
            self.get_parameter('overhead_camera.x').value,
            self.get_parameter('overhead_camera.y').value,
            self.get_parameter('overhead_camera.z').value,
        ])
        self.side_cam_pos = np.array([
            self.get_parameter('side_camera.x').value,
            self.get_parameter('side_camera.y').value,
            self.get_parameter('side_camera.z').value,
        ])

        # Table/cube geometry
        self.table_height = self.get_parameter('table_height').value
        self.cube_half = self.get_parameter('cube_half_height').value

        # Detection stability
        self.stability_readings = self.get_parameter('position_stability_readings').value
        self.stability_tol = self.get_parameter('position_stability_tolerance').value

    def _detect_colors(self, bgr_image):
        """
        Run the color segmentation pipeline on a BGR image.

        Returns a dict: color_name -> list of (cx, cy, area) centroid tuples
        in pixel coordinates.
        """
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        detections = {}

        # Morphological kernel for noise removal
        kernel = np.ones((5, 5), np.uint8)

        for color_name, params in self.hsv_ranges.items():
            # Build mask from all HSV ranges (handles red wrap-around)
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for (lower, upper) in params['ranges']:
                mask |= cv2.inRange(hsv, lower, upper)

            # Morphological open/close to remove noise
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
                # Keep the largest detection per color (we expect one cube per color)
                color_dets.sort(key=lambda d: d[2], reverse=True)
                detections[color_name] = color_dets[0]

        return detections

    def _annotate_image(self, bgr_image, detections, coord_text=None):
        """Draw bounding boxes and labels on the image."""
        annotated = bgr_image.copy()

        for color_name, (cx, cy, area) in detections.items():
            bgr_color = self.hsv_ranges[color_name]['bgr']
            # Draw crosshair at centroid
            cv2.circle(annotated, (cx, cy), 8, bgr_color, 2)
            cv2.line(annotated, (cx - 12, cy), (cx + 12, cy), bgr_color, 2)
            cv2.line(annotated, (cx, cy - 12), (cx, cy + 12), bgr_color, 2)

            # Label with color name
            label = color_name.upper()
            if coord_text and color_name in coord_text:
                label += f' {coord_text[color_name]}'
            cv2.putText(annotated, label, (cx + 10, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 2)

        return annotated

    def _overhead_back_project(self, cx_px, cy_px):
        """
        Back-project a pixel from the overhead (top-down) camera to world XY.

        The overhead camera is at self.overhead_cam_pos pointing straight down.
        We know the table surface Z, so we can solve for world X, Y.

        Camera convention:
        - Camera is rotated to point down (rpy=0, pi/2, 0 in URDF)
        - In camera frame: X=right, Y=down, Z=forward (toward table)
        """
        # Distance from camera to table surface (cube top)
        z_target = self.table_height + self.cube_half  # cube surface
        dist_to_target = self.overhead_cam_pos[2] - z_target

        # Back-project pixel to normalized camera coordinates
        nx = (cx_px - self.cx) / self.fx
        ny = (cy_px - self.cy) / self.fy

        # For overhead camera pointing down (-Z in world):
        # camera +X maps to world +X, camera +Y maps to world +Y (approximately)
        world_x = self.overhead_cam_pos[0] + nx * dist_to_target
        world_y = self.overhead_cam_pos[1] + ny * dist_to_target

        return world_x, world_y

    def _side_back_project_z(self, cy_px):
        """
        Back-project a pixel from the side camera to estimate world Z coordinate.

        The side camera is positioned to the side, looking horizontally.
        The vertical pixel position gives us the Z coordinate estimate.
        """
        # Distance from side camera to workspace center (approx)
        dist_to_workspace = abs(self.side_cam_pos[1])  # distance along Y

        # Back-project vertical pixel to get Z offset
        ny = (cy_px - self.cy) / self.fy

        # Side camera pointing horizontally: camera +Y maps to world -Z
        world_z = self.side_cam_pos[2] - ny * dist_to_workspace

        return world_z

    def _overhead_callback(self, msg):
        """Process overhead camera image: detect cubes and estimate XY positions."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error (overhead): {e}')
            return

        detections = self._detect_colors(bgr)

        # Back-project to world XY
        coord_text = {}
        for color_name, (cx, cy, area) in detections.items():
            world_x, world_y = self._overhead_back_project(cx, cy)
            self.overhead_detections[color_name] = (world_x, world_y)
            coord_text[color_name] = f'({world_x:.3f}, {world_y:.3f})'

        # Publish annotated image
        annotated = self._annotate_image(bgr, detections, coord_text)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.overhead_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'CV Bridge publish error: {e}')

    def _side_callback(self, msg):
        """Process side camera image: detect cubes and estimate Z positions."""
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error (side): {e}')
            return

        detections = self._detect_colors(bgr)

        # Back-project to world Z
        coord_text = {}
        for color_name, (cx, cy, area) in detections.items():
            world_z = self._side_back_project_z(cy)
            self.side_detections[color_name] = world_z
            coord_text[color_name] = f'(z={world_z:.3f})'

        # Publish annotated image
        annotated = self._annotate_image(bgr, detections, coord_text)
        try:
            out_msg = self.bridge.cv2_to_imgmsg(annotated, encoding='bgr8')
            out_msg.header = msg.header
            self.side_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().error(f'CV Bridge publish error: {e}')

    def _publish_fused_detections(self):
        """
        Fuse overhead (XY) and side (Z) detections into 3D poses.
        Publishes PoseArray with one Pose per detected cube.
        The color label is encoded in the orientation quaternion:
          red=(1,0,0,0), green=(0,1,0,0), yellow=(0,0,1,0)
        """
        pose_array = PoseArray()
        pose_array.header = Header()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = 'world'

        color_quat_map = {
            'red':    Quaternion(x=1.0, y=0.0, z=0.0, w=0.0),
            'green':  Quaternion(x=0.0, y=1.0, z=0.0, w=0.0),
            'yellow': Quaternion(x=0.0, y=0.0, z=1.0, w=0.0),
        }

        for color_name in ['red', 'green', 'yellow']:
            if color_name in self.overhead_detections:
                xy = self.overhead_detections[color_name]
                # Use side camera Z if available, otherwise default to table+cube height
                z = self.side_detections.get(
                    color_name,
                    self.table_height + self.cube_half
                )

                pose = Pose()
                pose.position = Point(x=xy[0], y=xy[1], z=z)
                pose.orientation = color_quat_map[color_name]

                # Track detection stability
                if color_name not in self.detection_history:
                    self.detection_history[color_name] = []
                self.detection_history[color_name].append((xy[0], xy[1], z))
                # Keep only recent readings
                if len(self.detection_history[color_name]) > self.stability_readings * 2:
                    self.detection_history[color_name] = \
                        self.detection_history[color_name][-self.stability_readings * 2:]

                pose_array.poses.append(pose)

        if pose_array.poses:
            self.detections_pub.publish(pose_array)

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
            pass  # TF not yet available, silently skip

    def is_position_stable(self, color_name):
        """
        Check if a cube's detected position is stable (consistent readings).

        Returns (stable: bool, position: tuple or None)
        """
        if color_name not in self.detection_history:
            return False, None

        history = self.detection_history[color_name]
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
        rclpy.shutdown()


if __name__ == '__main__':
    main()
