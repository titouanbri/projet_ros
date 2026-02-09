#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from tf2_msgs.msg import TFMessage
from geometry_msgs.msg import TransformStamped,Transform
import numpy as np
import cv2

def quaternion_to_rotation_matrix(qx,qy,qz,qw):
    """Convert quaternion to 3x3 rotation matrix."""
    n = np.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    qx,qy,qz,qw = qx/n, qy/n, qz/n, qw/n
    r00 = 1 - 2*(qy*qy + qz*qz)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx*qx + qz*qz)
    r12 = 2*(qy*qz - qx*qw)
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx*qx + qy*qy)
    return np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])

def rotation_matrix_to_quaternion(R):
    m00, m01, m02 = R[0]
    m10, m11, m12 = R[1]
    m20, m21, m22 = R[2]
    tr = m00 + m11 + m22
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (m21 - m12) / S
        qy = (m02 - m20) / S
        qz = (m10 - m01) / S
    elif m00 > m11 and m00 > m22:
        S = np.sqrt(1.0 + m00 - m11 - m22) * 2
        qw = (m21 - m12) / S
        qx = 0.25 * S
        qy = (m01 + m10) / S
        qz = (m02 + m20) / S
    elif m11 > m22:
        S = np.sqrt(1.0 + m11 - m00 - m22) * 2
        qw = (m02 - m20) / S
        qx = (m01 + m10) / S
        qy = 0.25 * S
        qz = (m12 + m21) / S
    else:
        S = np.sqrt(1.0 + m22 - m00 - m11) * 2
        qw = (m10 - m01) / S
        qx = (m02 + m20) / S
        qy = (m12 + m21) / S
        qz = 0.25 * S
    return np.array([qx, qy, qz, qw], dtype=np.float64)

def to_matrix(t : Transform):
    q = t.rotation
    trans = t.translation
    T = np.eye(4)
    T[:3,:3] = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
    T[0:3, 3] = [trans.x, trans.y, trans.z]
    return T

def from_matrix(M):
    t = TransformStamped()
    t.transform.translation.x = M[0,3]
    t.transform.translation.y = M[1,3]
    t.transform.translation.z = M[2,3]
    q = rotation_matrix_to_quaternion(M[:3,:3])
    t.transform.rotation.x = q[0]
    t.transform.rotation.y = q[1]
    t.transform.rotation.z = q[2]
    t.transform.rotation.w = q[3]
    return t

def hand_eye_solve(A_list, B_list):
    """
    A_list: list of 4x4 ee_i->ee_j
    B_list: list of 4x4 cam_i->cam_j
    Returns X: 4x4 cam->ee
    """
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for A,B in zip(A_list,B_list):
        R_gripper2base.append(A[:3,:3])
        t_gripper2base.append(A[:3,3].reshape(3,1))
        R_target2cam.append(B[:3,:3])
        t_target2cam.append(B[:3,3].reshape(3,1))

    R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
        R_gripper2base, t_gripper2base,
        R_target2cam, t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    X = np.eye(4)
    X[:3,:3] = R_cam2ee
    X[:3,3] = t_cam2ee.flatten()
    return X

def rotation_matrix_to_axis_angle(R):
    # Returns axis (3,) and angle in radians
    angle = np.arccos((np.trace(R)-1)/2)
    if abs(angle) < 1e-6:
        return np.array([1,0,0]),0
    rx = R[2,1]-R[1,2]
    ry = R[0,2]-R[2,0]
    rz = R[1,0]-R[0,1]
    axis = np.array([rx,ry,rz])
    axis = axis / np.linalg.norm(axis)
    return axis, angle


from tf2_ros import TransformBroadcaster

class HandEyeCalib(Node):

    def __init__(self):
        super().__init__('hand_eye_calib')
        self.ee_sub = self.create_subscription(TFMessage, '/ee_calib_poses', self.ee_cb, 10)
        self.aruco_sub = self.create_subscription(TFMessage, '/marker_calib_poses', self.aruco_cb, 10)

        self.tf_broadcaster = TransformBroadcaster(self)

        self.ee_latest = None
        self.aruco_latest = None
        self.ee_hist = []
        self.cam_hist = []
        self.samples_needed = 15
        self.calibrated = False
        self.T_base_cam = None

    def ee_cb(self, msg: TFMessage):
        if not msg.transforms:
            return
        self.ee_latest = msg.transforms[0]
        self.try_pair()

    def aruco_cb(self, msg: TFMessage):
        if not msg.transforms:
            return
        self.aruco_latest = msg.transforms[0]
        self.try_pair()

    def try_pair(self):
        if self.ee_latest and self.aruco_latest and not self.calibrated:
            ee = self.ee_latest
            ar = self.aruco_latest
            T_base_ee = to_matrix(ee.transform)
            T_cam_marker = to_matrix(ar.transform)
            T_marker_cam = np.linalg.inv(T_cam_marker)

            self.ee_hist.append(T_base_ee)
            self.cam_hist.append(T_marker_cam)

            self.ee_latest = None
            self.aruco_latest = None

            self.get_logger().info(f'Samples collected: {len(self.ee_hist)}/{self.samples_needed}')

            if len(self.ee_hist) >= self.samples_needed:
                self.calibrate()

    def calibrate(self):
        A_list = []
        B_list = []
        for i in range(len(self.ee_hist)-1):
            A1 = np.linalg.inv(self.ee_hist[i]) @ self.ee_hist[i+1]
            B1 = self.cam_hist[i] @ np.linalg.inv(self.cam_hist[i+1])
            A_list.append(A1)
            B_list.append(B1)

        X = hand_eye_solve(A_list, B_list)
        # Use first EE pose to compute camera w.r.t base
        self.T_base_cam = self.ee_hist[0] @ X
        self.calibrated = True
        self.get_logger().info('Hand-eye calibration done. Broadcasting camera TF.')

        # Start timer to broadcast continuously
        self.create_timer(0.1, self.broadcast_tf)

    def broadcast_tf(self):
        if self.T_base_cam is None:
            return
        t_msg = from_matrix(self.T_base_cam)
        t_msg.header.stamp = self.get_clock().now().to_msg()
        t_msg.header.frame_id = 'base_link'
        t_msg.child_frame_id = 'camera_link'
        self.tf_broadcaster.sendTransform(t_msg)

def main(args=None):
    rclpy.init(args=args)
    node = HandEyeCalib()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
