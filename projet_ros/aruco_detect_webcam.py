#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from rclpy.time import Time, Duration


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    n = 1.0 / math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx *= n; qy *= n; qz *= n; qw *= n
    qx2, qy2, qz2 = qx+qx, qy+qy, qz+qz
    xx, xy, xz = qx*qx2, qx*qy2, qx*qz2
    yy, yz, zz = qy*qy2, qy*qz2, qz*qz2
    wx, wy, wz = qw*qx2, qw*qy2, qw*qz2
    return np.array([
        [1-(yy+zz), xy-wz, xz+wy],
        [xy+wz, 1-(xx+zz), yz-wx],
        [xz-wy, yz+wx, 1-(xx+yy)]
    ], dtype=np.float64)


def rotation_matrix_to_quaternion(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = math.sqrt(tr+1.0)*2
        qw = 0.25*S
        qx = (m21-m12)/S
        qy = (m02-m20)/S
        qz = (m10-m01)/S
    elif m00>m11 and m00>m22:
        S = math.sqrt(1+m00-m11-m22)*2
        qw = (m21-m12)/S
        qx = 0.25*S
        qy = (m01+m10)/S
        qz = (m02+m20)/S
    elif m11>m22:
        S = math.sqrt(1+m11-m00-m22)*2
        qw = (m02-m20)/S
        qx = (m01+m10)/S
        qy = 0.25*S
        qz = (m12+m21)/S
    else:
        S = math.sqrt(1+m22-m00-m11)*2
        qw = (m10-m01)/S
        qx = (m02+m20)/S
        qy = (m12+m21)/S
        qz = 0.25*S
    return np.array([qx, qy, qz, qw], dtype=np.float64)


class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        # Aruco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, cv2.aruco.DetectorParameters())
        self.marker_length = 0.035
        s = self.marker_length/2.0
        self.marker_object_points = np.array([[-s,s,0],[s,s,0],[s,-s,0],[-s,-s,0]],dtype=np.float32)

        # Camera intrinsics
        fx = fy = 600.0
        cx = 320.0
        cy = 240.0
        self.camera_matrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],dtype=np.float64)
        self.dist_coeffs = np.zeros(5, dtype=np.float64)

        self.bridge = CvBridge()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.image_pub = self.create_publisher(Image, '/aruco/debug_image', 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)

        self.latest_tf = None
        self.cam_tf = None

        # Timer for TF broadcasting
        self.create_timer(0.01, self.broadcast_tf)

        # For smoothing
        self.prev_position = None
        self.alpha = 0.7  # simple exponential smoothing

    def rvec_to_quaternion(self, rvec):
        R, _ = cv2.Rodrigues(rvec)
        return rotation_matrix_to_quaternion(R)

    def broadcast_tf(self):
        if self.latest_tf:
            self.latest_tf.header.stamp = self.get_clock().now().to_msg()
            try:
                self.tf_broadcaster.sendTransform(self.latest_tf)
            except Exception as e:
                self.get_logger().warn(f"Failed to broadcast TF: {e}")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg,'bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            if ids is None:
                self.publish_image(msg, cv_image)
                return

            # Camera transform
            try:
                self.cam_tf = self.tf_buffer.lookup_transform(
                    'base_link','camera_link',Time(),timeout=Duration(seconds=0.01))
                R_cam = quaternion_to_rotation_matrix(
                    self.cam_tf.transform.rotation.x,
                    self.cam_tf.transform.rotation.y,
                    self.cam_tf.transform.rotation.z,
                    self.cam_tf.transform.rotation.w)
                t_cam = np.array([self.cam_tf.transform.translation.x,
                                  self.cam_tf.transform.translation.y,
                                  self.cam_tf.transform.translation.z],dtype=np.float64)
            except Exception:
                R_cam = np.eye(3)
                t_cam = np.zeros(3)

            for i in range(len(ids)):
                success, rvec, tvec = cv2.solvePnP(self.marker_object_points,
                                                  corners[i][0],
                                                  self.camera_matrix,
                                                  self.dist_coeffs,
                                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
                if not success:
                    continue

                # Smooth position
                if self.prev_position is None:
                    smooth_pos = tvec.flatten()
                else:
                    smooth_pos = self.alpha * self.prev_position + (1-self.alpha) * tvec.flatten()
                self.prev_position = smooth_pos

                # PoseStamped in camera frame
                pose = PoseStamped()
                pose.header.stamp = msg.header.stamp
                pose.header.frame_id = 'camera_link'
                pose.pose.position.x = smooth_pos[0]
                pose.pose.position.y = smooth_pos[1]
                pose.pose.position.z = smooth_pos[2]
                qx,qy,qz,qw = self.rvec_to_quaternion(rvec)
                pose.pose.orientation.x = qx
                pose.pose.orientation.y = qy
                pose.pose.orientation.z = qz
                pose.pose.orientation.w = qw
                self.pose_pub.publish(pose)

                # Transform to base_link
                x_aruco = R_cam @ smooth_pos.reshape(3,1) + t_cam.reshape(3,1)
                q = rotation_matrix_to_quaternion(R_cam @ quaternion_to_rotation_matrix(qx,qy,qz,qw))

                tf_msg = TransformStamped()
                tf_msg.header.stamp = msg.header.stamp
                tf_msg.header.frame_id = 'base_link'
                tf_msg.child_frame_id = 'aruco'
                tf_msg.transform.translation.x = float(x_aruco[0])
                tf_msg.transform.translation.y = float(x_aruco[1])
                tf_msg.transform.translation.z = float(x_aruco[2])
                tf_msg.transform.rotation.x = q[0]
                tf_msg.transform.rotation.y = q[1]
                tf_msg.transform.rotation.z = q[2]
                tf_msg.transform.rotation.w = q[3]
                self.latest_tf = tf_msg

                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvec, tvec, 0.03)

            cv2.aruco.drawDetectedMarkers(cv_image,corners,ids)
            self.publish_image(msg, cv_image)

        except Exception as e:
            self.get_logger().warn(f"Exception in image_callback: {e}")

    def publish_image(self, msg, image):
        try:
            out = self.bridge.cv2_to_imgmsg(image,'bgr8')
            out.header = msg.header
            self.image_pub.publish(out)
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
