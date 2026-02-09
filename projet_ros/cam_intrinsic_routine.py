#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

def quaternion_from_rpy(roll, pitch, yaw):
    """Return quaternion [x,y,z,w] from RPY (radians)."""
    cy = np.cos(yaw*0.5)
    sy = np.sin(yaw*0.5)
    cp = np.cos(pitch*0.5)
    sp = np.sin(pitch*0.5)
    cr = np.cos(roll*0.5)
    sr = np.sin(roll*0.5)
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    return np.array([x,y,z,w], dtype=np.float64)

def tilt_quaternion(base_quat, roll_offset=0, pitch_offset=0, yaw_offset=0):
    """Apply small RPY tilt to an existing quaternion (base_quat: [x,y,z,w])."""
    # Convert base to rotation matrix
    x,y,z,w = base_quat
    R = np.array([
        [1-2*(y**2+z**2), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x**2+z**2), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x**2+y**2)]
    ])

    # Small incremental rotation
    R_offset = np.eye(3)
    if roll_offset != 0:
        cr = np.cos(roll_offset)
        sr = np.sin(roll_offset)
        R_offset = R_offset @ np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    if pitch_offset != 0:
        cp = np.cos(pitch_offset)
        sp = np.sin(pitch_offset)
        R_offset = R_offset @ np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    if yaw_offset != 0:
        cy = np.cos(yaw_offset)
        sy = np.sin(yaw_offset)
        R_offset = R_offset @ np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])

    # Apply tilt relative to EE orientation
    R_new = R @ R_offset

    # Convert back to quaternion
    m00,m01,m02 = R_new[0]
    m10,m11,m12 = R_new[1]
    m20,m21,m22 = R_new[2]
    tr = m00+m11+m22
    if tr > 0:
        S = np.sqrt(tr+1.0)*2
        qw = 0.25*S
        qx = (m21-m12)/S
        qy = (m02-m20)/S
        qz = (m10-m01)/S
    elif m00>m11 and m00>m22:
        S = np.sqrt(1+m00-m11-m22)*2
        qw = (m21-m12)/S
        qx = 0.25*S
        qy = (m01+m10)/S
        qz = (m02+m20)/S
    elif m11>m22:
        S = np.sqrt(1+m11-m00-m22)*2
        qw = (m02-m20)/S
        qx = (m01+m10)/S
        qy = 0.25*S
        qz = (m12+m21)/S
    else:
        S = np.sqrt(1+m22-m00-m11)*2
        qw = (m10-m01)/S
        qx = (m02+m20)/S
        qy = (m12+m21)/S
        qz = 0.25*S
    return np.array([qx,qy,qz,qw], dtype=np.float64)


class CameraCalibTargets(Node):
    def __init__(self):
        super().__init__('camera_calib_targets')

        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- scheduling ---
        self.state = False
        self.create_subscription(Bool, '/camera_calib_cmd', self.cmd_cb, 1)
        self.feedback_pub = self.create_publisher(Bool, '/camera_calib_feedback', 1)
        self.capture_pub = self.create_publisher(Bool, '/camera_calib_capture', 1)

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_cb,
            1
        )

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        self.board_seen = False
        self.targets = []
        self.current_target = 0
        self.reach_thresh = 0.01

        self.create_timer(0.05, self.step)

    def cmd_cb(self, msg: Bool):
        self.state = msg.data
        if self.state:
            self.targets.clear()
            self.current_target = 0
            self.board_seen = False
            self.get_logger().info("Camera calibration motion started")

    def image_cb(self, msg: Image):
        if not self.state:
            return
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is not None and len(ids) > 6:
            self.board_seen = True

    def generate_targets(self, ee_tf):
        ee_pos = np.array([
            ee_tf.transform.translation.x,
            ee_tf.transform.translation.y,
            ee_tf.transform.translation.z
        ])

        offsets = [
            [0, 0, 0],
            [0.05, 0, 0],
            [-0.05, 0, 0],
            [0, 0.05, 0],
            [0, -0.10, 0],
            [0, 0, 0.08],
            [0.03, 0.03, 0.03],
            [-0.03, 0.03, 0.03],
            [0.03, -0.03, 0.05],
        ]

        for i, off in enumerate(offsets):
            tf = TransformStamped()
            tf.header.frame_id = 'base_link'
            tf.child_frame_id = f'cam_calib_target_{i}'
            tf.transform.translation.x = ee_pos[0] + off[0]
            tf.transform.translation.y = ee_pos[1] + off[1]
            tf.transform.translation.z = ee_pos[2] + off[2]

            roll_inc = np.deg2rad(1*i)
            pitch_inc = np.deg2rad(1*i)
            base_quat = np.array([
                ee_tf.transform.rotation.x,
                ee_tf.transform.rotation.y,
                ee_tf.transform.rotation.z,
                ee_tf.transform.rotation.w
            ])
            tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z, tf.transform.rotation.w = tilt_quaternion(base_quat, roll_inc, pitch_inc)
            self.targets.append(tf)

    def step(self):
        if not self.state:
            return

        try:
            ee_tf = self.tf_buffer.lookup_transform(
                'base_link',
                'wrist_3_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05)
            )
        except Exception:
            return

        if not self.targets and self.board_seen:
            self.generate_targets(ee_tf)

        if self.current_target >= len(self.targets):
            self.feedback_pub.publish(Bool(data=True))
            self.state = False
            self.get_logger().info("Camera calibration finished")
            return

        target = self.targets[self.current_target]
        target.header.stamp = self.get_clock().now().to_msg()
        target.child_frame_id = 'desired_ee'
        self.tf_broadcaster.sendTransform(target)

        ee_pos = np.array([
            ee_tf.transform.translation.x,
            ee_tf.transform.translation.y,
            ee_tf.transform.translation.z
        ])
        tgt_pos = np.array([
            target.transform.translation.x,
            target.transform.translation.y,
            target.transform.translation.z
        ])

        if np.linalg.norm(ee_pos - tgt_pos) < self.reach_thresh:
            self.capture_pub.publish(Bool(data=True))
            self.get_logger().info(f"Capture at target {self.current_target + 1}")
            self.current_target += 1


def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibTargets()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
