#!/usr/bin/env python3
# Broadcasts desired end-effector pose (wrt base_link) based on aruco marker detection

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
import numpy as np
import math

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion (x,y,z,w) to 3x3 rotation matrix."""
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    if norm == 0:
        return np.eye(3)
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    r00 = 1 - 2*(qy**2 + qz**2)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx**2 + qz**2)
    r12 = 2*(qy*qz - qx*qw)
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx**2 + qy**2)
    return np.array([[r00, r01, r02],
                     [r10, r11, r12],
                     [r20, r21, r22]])

def rotation_matrix_to_quaternion(R):
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
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
    elif (m00 > m11) and (m00 > m22):
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
    
    # Normalize
    norm = math.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
    return (qx/norm, qy/norm, qz/norm, qw/norm)

def RPY_to_R(roll, pitch, yaw):
    """Convert RPY (radians) to rotation matrix using ZYX convention."""
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll),  math.cos(roll)]])
    
    Ry = np.array([[ math.cos(pitch), 0, math.sin(pitch)],
                   [ 0,              1, 0             ],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw),  math.cos(yaw), 0],
                   [0,             0,              1]])
    
    return Rz @ Ry @ Rx  # ZYX order

class ArucoDesiredPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_desired_pose_node')

        # Declare parameters for desired pose offset (wrt aruco marker)
        self.declare_parameter('offset_x', 0.0)      # meters
        self.declare_parameter('offset_y', 0.0)      # meters
        self.declare_parameter('offset_z', 0.15)     # meters (15cm above marker)
        self.declare_parameter('roll', math.pi)      # radians
        self.declare_parameter('pitch', 0.0)         # radians
        self.declare_parameter('yaw', math.pi/2)     # radians
        self.declare_parameter('aruco_frame', 'aruco')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('desired_frame', 'desired_ee')
        self.declare_parameter('lookup_timeout', 0.1)  # seconds

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.timer = self.create_timer(0.01, self.timer_cb)
        self.get_logger().info("Aruco desired pose broadcaster initialized")

    def timer_cb(self):
        try:
            # Get parameters
            aruco_frame = self.get_parameter('aruco_frame').value
            base_frame = self.get_parameter('base_frame').value
            desired_frame = self.get_parameter('desired_frame').value
            timeout = rclpy.duration.Duration(seconds=self.get_parameter('lookup_timeout').value)

            # Lookup transform: base_link -> aruco (pose of aruco in base_link frame)
            try:
                marker_tf = self.tf_buffer.lookup_transform(
                    base_frame, aruco_frame, rclpy.time.Time(), timeout
                )
            except (LookupException, ConnectivityException, ExtrapolationException) as ex:
                self.get_logger().debug(f"Transform lookup failed: {ex}", throttle_duration_sec=1.0)
                return

            # Build homogeneous transform for aruco marker in base_link frame
            t = marker_tf.transform.translation
            q = marker_tf.transform.rotation
            R_marker = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)
            H_marker_to_base = np.eye(4)
            H_marker_to_base[:3, :3] = R_marker
            H_marker_to_base[:3, 3] = [t.x, t.y, t.z]

            # Build desired pose transform: aruco -> desired_ee
            H_desired_in_aruco = np.eye(4)
            H_desired_in_aruco[:3, :3] = RPY_to_R(
                self.get_parameter('roll').value,
                self.get_parameter('pitch').value,
                self.get_parameter('yaw').value
            )
            H_desired_in_aruco[:3, 3] = [
                self.get_parameter('offset_x').value,
                self.get_parameter('offset_y').value,
                self.get_parameter('offset_z').value
            ]

            # Compute desired pose in base_link frame: T_base->desired = T_base->aruco * T_aruco->desired
            H_desired_in_base = H_marker_to_base @ H_desired_in_aruco

            # Create and broadcast transform: base_link -> desired_ee
            tf_msg = TransformStamped()
            tf_msg.header.stamp = self.get_clock().now().to_msg()
            tf_msg.header.frame_id = base_frame
            tf_msg.child_frame_id = desired_frame
            
            tf_msg.transform.translation.x = H_desired_in_base[0, 3]
            tf_msg.transform.translation.y = H_desired_in_base[1, 3]
            tf_msg.transform.translation.z = H_desired_in_base[2, 3]
            
            qx, qy, qz, qw = rotation_matrix_to_quaternion(H_desired_in_base[:3, :3])
            tf_msg.transform.rotation.x = qx
            tf_msg.transform.rotation.y = qy
            tf_msg.transform.rotation.z = qz
            tf_msg.transform.rotation.w = qw

            self.tf_broadcaster.sendTransform(tf_msg)

        except Exception as e:
            self.get_logger().error(f"Error broadcasting desired pose: {e}", throttle_duration_sec=1.0)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDesiredPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()