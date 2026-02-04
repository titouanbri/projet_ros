
#broadcasts desired tf wrt aruco marker
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import numpy as np
import math

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
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
    return np.array([[r00,r01,r02],[r10,r11,r12],[r20,r21,r22]])

def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (x,y,z,w)."""
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
    return np.array([qx,qy,qz,qw], dtype=np.float64)

def tf_from_pose(H, parent_frame, child_frame, stamp):
    """Create a TransformStamped from a 4x4 pose."""
    t = H[:3,3]
    R = H[:3,:3]
    qx,qy,qz,qw = rotation_matrix_to_quaternion(R)
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = t[0]
    tf_msg.transform.translation.y = t[1]
    tf_msg.transform.translation.z = t[2]
    tf_msg.transform.rotation.x = qx
    tf_msg.transform.rotation.y = qy
    tf_msg.transform.rotation.z = qz
    tf_msg.transform.rotation.w = qw
    return tf_msg

def RPY_to_R(roll, pitch, yaw):
    """
    Converts roll, pitch, yaw angles to a 3x3 rotation matrix.
    Angles are in radians.

    Roll  = rotation about X
    Pitch = rotation about Y
    Yaw   = rotation about Z

    ZYX order (yaw-pitch-roll)
    """

    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])

    Ry = np.array([
        [ np.cos(pitch), 0, np.sin(pitch)],
        [ 0,             1, 0            ],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])

    # ZYX order: yaw → pitch → roll
    R = Rz @ Ry @ Rx
    return R

class ArucoDesiredPoseNode(Node):
    def __init__(self):
        super().__init__('aruco_desired_pose_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.marker_tf = None
        self.timer = self.create_timer(0.01, self.timer_cb)

    def timer_cb(self):
        try:
            aruco_frame = "aruco"
            base_frame = "base_link"

            # Lookup the ARUCO marker pose wrt base
            if self.tf_buffer.can_transform(base_frame, aruco_frame, rclpy.time.Time()):
                self.marker_tf = self.tf_buffer.lookup_transform(
                    base_frame, aruco_frame, rclpy.time.Time()
                )

                # Define desired pose wrt ARUCO (e.g., 5cm above)
                H_desired = np.eye(4)
                H_desired[:3,:3] = RPY_to_R(np.pi,0,np.pi/2)
                H_desired[:3,3] = np.array([0,0,0.15])  # 5cm offset along marker z

                t_marker = self.marker_tf.transform.translation
                q_marker = self.marker_tf.transform.rotation
                R_marker = quaternion_to_rotation_matrix(
                    q_marker.x, q_marker.y, q_marker.z, q_marker.w
                )
                H_marker = np.block([[R_marker, np.array([t_marker.x, t_marker.y, t_marker.z]).reshape(3,1)],
                                     [np.zeros((1,3)),1]])

                # Desired pose in world/base frame
                H_desired_world = H_marker @ H_desired

                # Broadcast it
                tf_msg = tf_from_pose(H_desired_world, parent_frame=base_frame,
                                      child_frame="desired_pose", stamp=self.get_clock().now().to_msg())
                self.tf_broadcaster.sendTransform(tf_msg)

        except Exception as e:
            self.get_logger().warn(f"Failed to define desired pose: {e}", throttle_duration_sec=5.0)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDesiredPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
