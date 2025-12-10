#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Imu, Image, PointCloud2, PointField
from geometry_msgs.msg import Twist, Point32, PointStamped, PolygonStamped, PoseStamped, PoseArray
from nav_msgs.msg import Path
from std_msgs.msg import Header
from rclpy.qos import qos_profile_sensor_data, QoSProfile, ReliabilityPolicy, DurabilityPolicy
import numpy as np
import onnxruntime as ort
import os
from ament_index_python.packages import get_package_share_directory
from shapely.geometry import Polygon
from shapely.ops import polylabel
from collections import deque
import struct

# --- CONFIGURATION ---
TOP_LIDAR_TOPIC = '/X3/lidar/top/scan'
BOT_LIDAR_TOPIC = '/X3/lidar/bottom/scan'
IMU_TOPIC = '/X3/imu'
CMD_TOPIC = '/X3/cmd_vel'
DEBUG_IMG_TOPIC = '/cnn_navigator/debug_image'
DEBUG_POLY_TOPIC = '/cnn_navigator/debug_poly'
DEBUG_CENTER_TOPIC = '/cnn_navigator/debug_center'
POSE_TOPIC = '/model/X3/pose'
TUNNEL_MAP_TOPIC = '/tunnel_viz/tunnel_map'
DRONE_PATH_TOPIC = '/tunnel_viz/drone_path'

# Tunnel map settings
MAX_TUNNEL_SLICES = 200  # Maximum number of cross-sections to keep
# Minimum movement (meters) before adding new slice
MIN_MOVEMENT_FOR_SLICE = 0.5
MAX_RANGE = 10.0  # Maximum LiDAR range to consider

# CNN Constants
LIDAR_FOV_V = 45.0 * (np.pi / 180.0)
IMG_H, IMG_W = 60, 80
MAX_Y = 5.0
MAX_Z = 4.0
MAX_X = 1.5


class CNNNavigator(Node):
    def __init__(self):
        super().__init__('cnn_navigator')

        # 1. Load ONNX Model
        pkg_share = get_package_share_directory('tunnel_pilot')
        model_path = os.path.join(pkg_share, 'models', 'tunnel_cnn.onnx')

        try:
            self.session = ort.InferenceSession(
                model_path, providers=['CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.get_logger().info('Brain Loaded Successfully!')
        except Exception as e:
            self.get_logger().error(f'Model Load Failed: {e}')
            return

        # 2. State Variables
        self.scan_top_msg = None
        self.scan_bot_msg = None
        self.current_roll = 0.0
        self.current_pitch = 0.0

        # --- NEW: Tunnel map and path state ---
        self.drone_position = np.array([0.0, 0.0, 0.0])
        self.drone_yaw = 0.0
        self.last_slice_position = np.array([0.0, 0.0, 0.0])
        self.tunnel_history = deque(maxlen=MAX_TUNNEL_SLICES)
        self.path_history = []
        self.max_path_points = 1000

        # 3. Subscribers
        self.create_subscription(
            LaserScan, TOP_LIDAR_TOPIC, self.top_cb, qos_profile_sensor_data)
        self.create_subscription(
            LaserScan, BOT_LIDAR_TOPIC, self.bot_cb, qos_profile_sensor_data)
        self.create_subscription(
            Imu, IMU_TOPIC, self.imu_cb, qos_profile_sensor_data)

        # --- NEW: Pose subscriber for tunnel map ---
        pose_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=1
        )
        self.create_subscription(
            PoseArray, POSE_TOPIC, self.pose_cb, pose_qos)

        # 4. Publishers
        self.cmd_pub = self.create_publisher(Twist, CMD_TOPIC, 10)
        self.debug_pub = self.create_publisher(Image, DEBUG_IMG_TOPIC, 10)

        # --- NEW DEBUG PUBLISHERS ---
        self.poly_pub = self.create_publisher(
            PolygonStamped, DEBUG_POLY_TOPIC, 10)
        self.center_pub = self.create_publisher(
            PointStamped, DEBUG_CENTER_TOPIC, 10)

        # --- NEW: Tunnel map and path publishers ---
        self.tunnel_map_pub = self.create_publisher(
            PointCloud2, TUNNEL_MAP_TOPIC, 10)
        self.path_pub = self.create_publisher(
            Path, DRONE_PATH_TOPIC, 10)

        # 5. Control Timer (20Hz)
        self.create_timer(0.05, self.control_loop)

        # --- NEW: Tunnel map publish timer (5Hz) ---
        self.create_timer(0.2, self.publish_tunnel_map)

        self.get_logger().info('Tunnel Map and Path publishers initialized!')
        self.get_logger().info(f'  - Tunnel Map: {TUNNEL_MAP_TOPIC}')
        self.get_logger().info(f'  - Drone Path: {DRONE_PATH_TOPIC}')

    def top_cb(self, msg):
        self.scan_top_msg = msg

    def bot_cb(self, msg):
        self.scan_bot_msg = msg

    def imu_cb(self, msg):
        q = msg.orientation
        sinr_cosp = 2 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1 - 2 * (q.x * q.x + q.y * q.y)
        self.current_roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self.current_pitch = np.sign(sinp) * (np.pi / 2)
        else:
            self.current_pitch = np.arcsin(sinp)

        # Extract yaw for tunnel map
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.drone_yaw = np.arctan2(siny_cosp, cosy_cosp)

    # --- NEW: Pose callback for tunnel map ---
    def pose_cb(self, msg):
        """Get drone position from Gazebo pose array."""
        if len(msg.poses) > 0:
            # Last pose in array is usually the model pose
            pose = msg.poses[-1]
            self.drone_position = np.array([
                pose.position.x,
                pose.position.y,
                pose.position.z
            ])

            # Update path history
            if len(self.path_history) == 0 or \
               np.linalg.norm(self.drone_position - np.array([
                   self.path_history[-1].pose.position.x,
                   self.path_history[-1].pose.position.y,
                   self.path_history[-1].pose.position.z
               ])) > 0.1:

                pose_stamped = PoseStamped()
                pose_stamped.header.stamp = self.get_clock().now().to_msg()
                pose_stamped.header.frame_id = "world"
                pose_stamped.pose = pose
                self.path_history.append(pose_stamped)

                # Limit path history size
                if len(self.path_history) > self.max_path_points:
                    self.path_history = self.path_history[-self.max_path_points:]

                # Publish path
                self.publish_drone_path()

    def publish_drone_path(self):
        """Publish the drone's flight path."""
        path_msg = Path()
        path_msg.header.stamp = self.get_clock().now().to_msg()
        path_msg.header.frame_id = "world"
        path_msg.poses = self.path_history
        self.path_pub.publish(path_msg)

    def add_tunnel_slice(self, y_points, z_points):
        """Add a new cross-section slice to the tunnel history."""
        # Check if we've moved enough since last slice
        dist = np.linalg.norm(self.drone_position - self.last_slice_position)
        if dist < MIN_MOVEMENT_FOR_SLICE and len(self.tunnel_history) > 0:
            return

        # Store the slice with world position and orientation
        self.tunnel_history.append({
            'position': self.drone_position.copy(),
            'yaw': self.drone_yaw,
            'y_points': y_points.copy(),
            'z_points': z_points.copy()
        })
        self.last_slice_position = self.drone_position.copy()

    def publish_tunnel_map(self):
        """Publish accumulated tunnel cross-sections as a 3D point cloud."""
        if len(self.tunnel_history) == 0:
            return

        all_points = []

        for slice_data in self.tunnel_history:
            pos = slice_data['position']
            yaw = slice_data['yaw']
            y_pts = slice_data['y_points']
            z_pts = slice_data['z_points']

            # Rotation matrix for yaw
            cos_yaw = np.cos(yaw)
            sin_yaw = np.sin(yaw)

            # Transform local y-z points to world coordinates
            for i in range(len(y_pts)):
                # Local point (0, y, z) -> rotate by yaw -> translate by position
                local_y = y_pts[i]
                local_z = -z_pts[i]  # Flip Z to correct orientation

                # Rotate around Z axis (yaw) - standard rotation matrix
                world_x = pos[0] + local_y * sin_yaw
                world_y = pos[1] + local_y * cos_yaw
                world_z = pos[2] + local_z

                # Color based on height
                normalized_z = np.clip((local_z + 3) / 6, 0, 1)
                if normalized_z < 0.5:
                    r = int(2 * normalized_z * 255)
                    g = 255
                    b = int((1 - 2 * normalized_z) * 255)
                else:
                    r = 255
                    g = int((2 - 2 * normalized_z) * 255)
                    b = 0

                rgb = (r << 16) | (g << 8) | b
                all_points.append((world_x, world_y, world_z, rgb))

        if len(all_points) == 0:
            return

        # Create PointCloud2 message
        header = Header()
        header.stamp = self.get_clock().now().to_msg()
        header.frame_id = "world"

        fields = [
            PointField(name='x', offset=0,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,
                       datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12,
                       datatype=PointField.UINT32, count=1),
        ]

        points_data = []
        for pt in all_points:
            points_data.append(struct.pack('fffI', pt[0], pt[1], pt[2], pt[3]))

        msg = PointCloud2()
        msg.header = header
        msg.height = 1
        msg.width = len(all_points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(all_points)
        msg.is_dense = True
        msg.data = b''.join(points_data)

        self.tunnel_map_pub.publish(msg)

    def resize_scan(self, ranges, target_size=360):
        ranges = np.array(ranges)
        ranges[np.isinf(ranges)] = 10.0
        ranges[np.isnan(ranges)] = 10.0
        x_old = np.linspace(0, 1, len(ranges))
        x_new = np.linspace(0, 1, target_size)
        return np.interp(x_new, x_old, ranges).astype(np.float32)

    def find_safest_spot(self, y_points, z_points):
        """
        Finds the Pole of Inaccessibility AND publishes visualization.
        """
        if len(y_points) < 3:
            return 0.0, 0.0, 0.0

        # 1. Create Polygon
        angles = np.arctan2(z_points, y_points)
        sorted_indices = np.argsort(angles)
        poly_points = np.column_stack(
            (y_points[sorted_indices], z_points[sorted_indices]))

        try:
            poly = Polygon(poly_points)
            if not poly.is_valid:
                return 0.0, 0.0, 0.0

            # 2. Simplify Shape (RDP Algorithm)
            simplified_poly = poly.simplify(0.1, preserve_topology=False)

            # 3. Find Pole of Inaccessibility
            best_point = polylabel(simplified_poly, tolerance=0.1)
            radius = simplified_poly.exterior.distance(best_point)

            # --- VISUALIZATION LOGIC ---
            self.publish_viz(simplified_poly, best_point)

            return best_point.x, best_point.y, radius

        except Exception as e:
            return 0.0, 0.0, 0.0

    def publish_viz(self, poly, center_point):

        # 1. Publish Center Point (Red Dot)
        pt_msg = PointStamped()
        pt_msg.header.stamp = self.get_clock().now().to_msg()
        pt_msg.header.frame_id = "base_link"  # Shows up relative to drone
        pt_msg.point.y = center_point.x  # Y in 2D projection is Y in ROS
        pt_msg.point.z = center_point.y  # Z in 2D projection is Z in ROS
        pt_msg.point.x = 2.0  # Push it 2m forward so you can see it in RViz
        self.center_pub.publish(pt_msg)

        # 2. Publish Polygon (Green Outline)
        poly_msg = PolygonStamped()
        poly_msg.header = pt_msg.header

        # Extract coordinates from shapely
        coords = list(poly.exterior.coords)
        for p in coords:
            ros_pt = Point32()
            ros_pt.x = 2.0  # Push forward 2m
            ros_pt.y = float(p[0])
            ros_pt.z = float(p[1])
            poly_msg.polygon.points.append(ros_pt)

        self.poly_pub.publish(poly_msg)

    def preprocess(self, ranges_top, ranges_bot, roll, pitch, angles):
        x_t = ranges_top * np.cos(angles) * np.cos(LIDAR_FOV_V)
        y_t = ranges_top * np.sin(angles)
        z_t = ranges_top * np.cos(angles) * np.sin(LIDAR_FOV_V)

        x_b = ranges_bot * np.cos(angles) * np.cos(-LIDAR_FOV_V)
        y_b = ranges_bot * np.sin(angles)
        z_b = ranges_bot * np.cos(angles) * np.sin(-LIDAR_FOV_V)

        x_final = np.concatenate([x_t, x_b])
        y = np.concatenate([y_t, y_b])
        z = np.concatenate([z_t, z_b])

        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        x_p = x_final * cp + z * sp
        z_p = -x_final * sp + z * cp
        y_final = y * cr - z_p * sr
        z_final = y * sr + z_p * cr

        safe_y, safe_z, safe_radius = self.find_safest_spot(y_final, z_final)

        # Subsample points for efficiency
        step = max(1, len(y_final) // 100)
        self.add_tunnel_slice(y_final[::step], z_final[::step])

        # Fix inverted projection
        u = ((-y_final / MAX_Y) * (IMG_W / 2) + (IMG_W / 2)).astype(int)
        v = ((z_final / MAX_Z) * (IMG_H / 2) + (IMG_H / 2)).astype(int)

        img = np.zeros((3, IMG_H, IMG_W), dtype=np.float32)
        valid = (u >= 0) & (u < IMG_W) & (v >= 0) & (v < IMG_H)
        u, v = u[valid], v[valid]

        pitch_normalized = np.clip((pitch + 0.35) / 0.7, 0, 1)
        roll_normalized = np.clip((roll + 0.35) / 0.7, 0, 1)

        # R = pitch
        # G = roll
        # B = depth
        img[0, v, u] = pitch_normalized
        img[1, v, u] = roll_normalized
        img[2, v, u] = np.clip(x_p[valid] / MAX_X, 0, 1)

        return np.expand_dims(img, axis=0), safe_y, safe_z, safe_radius

    def publish_debug_image(self, input_tensor):
        img_array = input_tensor[0]

        # DEBUG: Print channel values
        mask = img_array[0] > 0  # Non-zero pixels
        if mask.any():
            print(f"R (pitch): {img_array[0][mask][0]:.3f}")
            print(f"G (roll):  {img_array[1][mask][0]:.3f}")
            print(
                f"B (depth): min={img_array[2][mask].min():.3f}, max={img_array[2][mask].max():.3f}")

        img_display = np.transpose(img_array, (1, 2, 0))
        img_uint8 = (np.clip(img_display, 0, 1) * 255).astype(np.uint8)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.height = IMG_H
        msg.width = IMG_W
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = IMG_W * 3
        msg.data = img_uint8.tobytes()
        self.debug_pub.publish(msg)

    def control_loop(self):
        if self.scan_top_msg is None or self.scan_bot_msg is None:
            self.get_logger().warning('Waiting for LiDARs...', throttle_duration_sec=2.0)
            return

        r_top = self.resize_scan(self.scan_top_msg.ranges, 360)
        r_bot = self.resize_scan(self.scan_bot_msg.ranges, 360)
        angles = np.linspace(-np.pi, np.pi, 360)

        input_tensor, safe_y, safe_z, safe_radius = self.preprocess(
            r_top, r_bot, self.current_roll, self.current_pitch, angles)

        self.publish_debug_image(input_tensor)

        outputs = self.session.run(None, {self.input_name: input_tensor})
        predicted_yaw = outputs[0][0][0]
        print(predicted_yaw)

        twist = Twist()

        # twist.angular.z = float(predicted_yaw) * 0.25

        twist.linear.y = float(safe_y) * -0.1
        twist.linear.z = float(safe_z) * -0.1

        # 3. SPEED
        target_speed = np.clip(safe_radius * 0.5, 0.5, 6.0)
        twist.linear.x = float(target_speed)

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = CNNNavigator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
