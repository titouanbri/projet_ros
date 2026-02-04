#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import TransformStamped, Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import TransformBroadcaster, Buffer, TransformListener,TFMessage


def rotation_matrix_to_quaternion(R):
    """ x,y,z,w"""
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = math.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = math.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = math.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def quat_from_z_towards_target(pos_child, pos_target):
    z = pos_target - pos_child
    z /= np.linalg.norm(z)
    x_ref = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(z, x_ref)) > 0.99:
        x_ref = np.array([0.0, 1.0, 0.0])
    x = np.cross(x_ref, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    return rotation_matrix_to_quaternion(R)


class EETargetsNode(Node):
    def __init__(self):
        super().__init__('ee_targets_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.bridge = CvBridge()

        self.marker_detected = False
        self.targets_generated = False
        self.corners = None

        self.target_tfs = []
        self.current_target_idx = 0
        self.reach_thresh = 0.01

        self.ee_calib_list = []
        self.marker_calib_list = []

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        self.marker_length = 0.035
        s = self.marker_length / 2.0
        self.marker_object_points = np.array(
            [[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]],
            dtype=np.float32
        )

        self.camera_matrix = None
        self.dist_coeffs = None

        self.debug_pub = self.create_publisher(Twist, '/debug_calib', 1)
        self.ee_calib_pub = self.create_publisher(TFMessage, '/ee_calib_poses', 10)
        self.marker_calib_pub = self.create_publisher(TFMessage, '/marker_calib_poses', 10)

        self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.camera_info_cb,
            1
        )
        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_cb,
            1
        )

        self.create_timer(0.05, self.step)

    def camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)

    def image_cb(self, msg):
        if self.camera_matrix is None:
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)

        if ids is not None:
            self.corners = corners
            if not self.marker_detected:
                self.marker_detected = True
                self.marker_first_seen = self.get_clock().now()
                self.get_logger().info(
                    "Marker detected, waiting 2 seconds before generating targets"
                )

    def step(self):
        # generate targets once
        if self.marker_detected and not self.targets_generated:

            elapsed = (
                self.get_clock().now() - self.marker_first_seen
            ).nanoseconds / 1e9

            if elapsed < 2.0:
                return

            try:
                ee_tf = self.tf_buffer.lookup_transform(
                    'base_link',
                    'wrist_3_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except Exception:
                return

            ee_pos = np.array([
                ee_tf.transform.translation.x,
                ee_tf.transform.translation.y,
                ee_tf.transform.translation.z
            ])

            offsets = [
                np.array([0.05, 0, 0]),
                np.array([-0.05, 0, 0]),
                np.array([0, 0, 0.1]),
                np.array([0, 0.01, 0.05]),
                np.array([-0.02, 0.01, 0.05]),
                np.array([0.02, 0.02, 0.02]),

            ]

            marker_pos_base = ee_pos + np.array([0.5, 0, 0])
            self.target_tfs.clear()

            for i, off in enumerate(offsets):
                pos = ee_pos + off
                #q = quat_from_z_towards_target(pos, marker_pos_base)
                

                tf_msg = TransformStamped()
                tf_msg.header.frame_id = 'base_link'
                tf_msg.child_frame_id = f'calib_{i+1}'
                tf_msg.transform.translation.x = float(pos[0])
                tf_msg.transform.translation.y = float(pos[1])
                tf_msg.transform.translation.z = float(pos[2])
                #tf_msg.transform.rotation.x = float(q[0])
                #tf_msg.transform.rotation.y = float(q[1])
                #tf_msg.transform.rotation.z = float(q[2])
                #tf_msg.transform.rotation.w = float(q[3])
                tf_msg.transform.rotation = ee_tf.transform.rotation

                self.target_tfs.append(tf_msg)

            self.targets_generated = True
            self.get_logger().info("Target TFs generated")

        # execute targets
        if self.targets_generated and self.current_target_idx < len(self.target_tfs):
            target_tf = self.target_tfs[self.current_target_idx]
            target_tf.header.stamp = self.get_clock().now().to_msg()
            target_tf.child_frame_id = 'desired_ee'
            self.tf_broadcaster.sendTransform(target_tf)

            try:
                ee_tf = self.tf_buffer.lookup_transform(
                    'base_link',
                    'wrist_3_link',
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.01)
                )
            except Exception as e:
                self.get_logger().warn(f"Error getting EE TF : {e}")
                return

            ee_pos = np.array([
                ee_tf.transform.translation.x,
                ee_tf.transform.translation.y,
                ee_tf.transform.translation.z
            ])
            target_pos = np.array([
                target_tf.transform.translation.x,
                target_tf.transform.translation.y,
                target_tf.transform.translation.z
            ])

            if np.linalg.norm(ee_pos - target_pos) < self.reach_thresh:
                try:
                    current_ee = self.tf_buffer.lookup_transform(
                        'base_link',
                        'wrist_3_link',
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.01)
                    )
                except Exception as e:
                    self.get_logger().warn(f"Error getting EE TF : {e}")
                    return
                
                if self.corners is None:
                    return

                success, rvec, tvec = cv2.solvePnP(
                    self.marker_object_points,
                    self.corners[0][0],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    tvec = tvec.flatten()
                    rvec = rvec.flatten()

                    R_marker,_ = cv2.Rodrigues(rvec)

                    q = rotation_matrix_to_quaternion(R_marker)

                    marker_tf = TransformStamped()

                    marker_tf.transform.translation.x = tvec[0]
                    marker_tf.transform.translation.y = tvec[1]
                    marker_tf.transform.translation.z = tvec[2]

                    marker_tf.transform.rotation.x = q[0]
                    marker_tf.transform.rotation.y = q[1]
                    marker_tf.transform.rotation.z = q[2]
                    marker_tf.transform.rotation.w = q[3]

                    self.marker_calib_list.append(marker_tf)
                    self.ee_calib_list.append(current_ee)


                    msg = Twist()
                    msg.linear.x, msg.linear.y, msg.linear.z = tvec
                    msg.angular.x, msg.angular.y, msg.angular.z = rvec
                    self.debug_pub.publish(msg)

                    self.get_logger().info(
                        f"PnP success at target {self.current_target_idx + 1}"
                    )

                    # allow fresh detection for next pose
                    self.marker_detected = False
                    self.corners = None
                    self.current_target_idx += 1

                else:
                    self.get_logger().warn("solvePnP failed")

        if len(self.marker_calib_list) == 6 and len(self.marker_calib_list) == 6:
            ee_calib = TFMessage()
            ee_calib.transforms = self.ee_calib_list
            self.ee_calib_pub.publish(ee_calib)
            marker_calib = TFMessage()
            marker_calib.transforms = self.marker_calib_list
            self.marker_calib_pub.publish(marker_calib)
        elif self.current_target_idx == 5:
            self.get_logger().warn("Couldn't get full calibration info")

def main(args=None):
    rclpy.init(args=args)
    node = EETargetsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
