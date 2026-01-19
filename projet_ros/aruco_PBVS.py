import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import Buffer, TransformListener
import numpy as np
from scipy.linalg import logm

def hat_to_twist(xi_hat: np.ndarray):
    """Convert se(3) matrix to 6x1 twist vector (angular | linear)."""
    w = np.array([xi_hat[2,1], xi_hat[0,2], xi_hat[1,0]])
    v = xi_hat[:3,3]
    return np.hstack((w,v))

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

def pose_to_matrix(tf: TransformStamped):
    """Convert TransformStamped to 4x4 homogeneous matrix."""
    t = tf.transform.translation
    q = tf.transform.rotation
    R = quaternion_to_rotation_matrix(q.x,q.y,q.z,q.w)
    x = np.array([t.x,t.y,t.z])
    H = np.block([[R, x.reshape(3,1)],
                  [np.zeros((1,3)),1]])
    return H

def adjoint(H: np.ndarray):
    """Adjoint of 4x4 homogeneous matrix."""
    R = H[:3,:3]
    p = H[:3,3]
    adj = np.block([[R, np.zeros((3,3))],
                    [skew(p)@R, R]])
    return adj

def skew(v):
    v = np.array(v).flatten()
    return np.array([[0,-v[2],v[1]],
                     [v[2],0,-v[0]],
                     [-v[1],v[0],0]])

class EEPositionController(Node):
    def __init__(self):
        super().__init__('ee_position_controller')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cmd_pub = self.create_publisher(Twist,'/ee_velocity_cmd',1)

        self.dt = 0.01
        self.timer = self.create_timer(self.dt, self.timer_cb)

    def timer_cb(self):
        try:
            base_frame = "base_link"
            ee_frame = "wrist_3_link"
            desired_frame = "desired_pose"

            if self.tf_buffer.can_transform(base_frame, ee_frame, rclpy.time.Time()) and \
               self.tf_buffer.can_transform(base_frame, desired_frame, rclpy.time.Time()):

                ee_tf = self.tf_buffer.lookup_transform(base_frame, ee_frame, rclpy.time.Time())
                desired_tf = self.tf_buffer.lookup_transform(base_frame, desired_frame, rclpy.time.Time())

                self.compute_and_publish_twist(ee_tf, desired_tf)

        except Exception as e:
            self.get_logger().warn(f"Failed to get TFs: {e}", throttle_duration_sec=5.0)

    def compute_and_publish_twist(self, ee_tf: TransformStamped, desired_tf: TransformStamped):
        Kp = 1.0 *1e-2

        H_ee = pose_to_matrix(ee_tf)
        H_d  = pose_to_matrix(desired_tf)

        # error in se(3)
        xi_hat = logm(np.linalg.inv(H_ee) @ H_d)
        xi = hat_to_twist(xi_hat)

        # Control twist in world frame
        twist_vec = Kp * xi

        # Publish
        twist_msg = Twist()
        twist_msg.angular.x = float(twist_vec[0])
        twist_msg.angular.y = float(twist_vec[1])
        twist_msg.angular.z = float(twist_vec[2])
        twist_msg.linear.x = float(twist_vec[3])
        twist_msg.linear.y = float(twist_vec[4])
        twist_msg.linear.z = float(twist_vec[5])
        self.cmd_pub.publish(twist_msg)

def main(args=None):
    rclpy.init(args=args)
    node = EEPositionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
