#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import TransformBroadcaster, Buffer, TransformListener

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    n = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if n == 0.0:
        return np.eye(3)
    qx /= n; qy /= n; qz /= n; qw /= n
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1-2*(yy+zz), 2*(xy-wz), 2*(xz+wy)],
        [2*(xy+wz), 1-2*(xx+zz), 2*(yz-wx)],
        [2*(xz-wy), 2*(yz+wx), 1-2*(xx+yy)]
    ], dtype=np.float64)

def rotation_matrix_to_quaternion(R):
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

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        # ArUco detector setup
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())

        self.marker_length = 0.035
        s = self.marker_length / 2.0
        self.marker_object_points = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]], dtype=np.float32)

        # Dynamic intrinsics
        self.camera_matrix = None
        self.dist_coeffs = None
        self.intrinsics_warning_logged = False

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)

        # Subscriptions
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 1)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        self.latest_tf = None
        self.create_timer(0.01, self.broadcast_tf)

    def camera_info_callback(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k, dtype=np.float64).reshape(3,3)
        self.dist_coeffs = np.array(msg.d, dtype=np.float64)
        self.intrinsics_warning_logged = True

    def rvec_to_quaternion(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        return rotation_matrix_to_quaternion(R)

    def broadcast_tf(self):
        if self.latest_tf:
            self.latest_tf.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(self.latest_tf)

    def image_callback(self, msg):
        if self.camera_matrix is None:
            if not self.intrinsics_warning_logged:
                self.get_logger().warn("Camera intrinsics not yet received, skipping frame")
                self.intrinsics_warning_logged = True
            return

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn(f"CV bridge failed: {e}")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.aruco_detector.detectMarkers(gray)
        if ids is None:
            self.publish_image(msg, cv_image)
            return

        try:
            cam_tf = self.tf_buffer.lookup_transform(
                'base_link', 'camera_color_frame', # adjust to your camera frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.01)
            )
            R_cam = quaternion_to_rotation_matrix(
                cam_tf.transform.rotation.x,
                cam_tf.transform.rotation.y,
                cam_tf.transform.rotation.z,
                cam_tf.transform.rotation.w
            )
            t_cam = np.array([
                cam_tf.transform.translation.x,
                cam_tf.transform.translation.y,
                cam_tf.transform.translation.z
            ], dtype=np.float64)
        except Exception:
            R_cam = np.eye(3)
            t_cam = np.zeros(3)

        for i in range(len(ids)):
            try:
                success, rvec, tvec = cv2.solvePnP(
                    self.marker_object_points,
                    corners[i][0],
                    self.camera_matrix,
                    self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if not success:
                    continue

                tvec = tvec.flatten()
                if np.isnan(tvec).any():
                    continue

                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = 'camera_link'
                pose.pose.position.x = float(tvec[0])
                pose.pose.position.y = float(tvec[1])
                pose.pose.position.z = float(tvec[2])
                qx,qy,qz,qw = self.rvec_to_quaternion(rvec)
                pose.pose.orientation.x = float(qx)
                pose.pose.orientation.y = float(qy)
                pose.pose.orientation.z = float(qz)
                pose.pose.orientation.w = float(qw)
                self.pose_pub.publish(pose)

                # Transform to base_link
                t_aruco_base = (R_cam @ tvec.reshape(3,1) + t_cam.reshape(3,1)).ravel()
                R_aruco_base = R_cam @ quaternion_to_rotation_matrix(qx,qy,qz,qw)
                q_base = rotation_matrix_to_quaternion(R_aruco_base)

                tf_msg = TransformStamped()
                tf_msg.header.stamp = msg.header.stamp
                tf_msg.header.frame_id = 'base_link'
                tf_msg.child_frame_id = 'aruco'
                tf_msg.transform.translation.x = float(t_aruco_base[0])
                tf_msg.transform.translation.y = float(t_aruco_base[1])
                tf_msg.transform.translation.z = float(t_aruco_base[2])
                tf_msg.transform.rotation.x = float(q_base[0])
                tf_msg.transform.rotation.y = float(q_base[1])
                tf_msg.transform.rotation.z = float(q_base[2])
                tf_msg.transform.rotation.w = float(q_base[3])
                self.latest_tf = tf_msg

                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)
            except Exception as e:
                self.get_logger().warn(f"Aruco processing error: {e}")

        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)
        self.publish_image(msg, cv_image)

    def publish_image(self, msg, image):
        try:
            out_msg = self.bridge.cv2_to_imgmsg(image,'bgr8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ArucoTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
