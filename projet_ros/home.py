
#agnostic pose control
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from std_msgs.msg import Bool
import tf2_ros
import numpy as np
from scipy.linalg import logm

def clamp(v: np.ndarray, limit: float):
    v_norm = np.linalg.norm(v)
    if v_norm > limit:  
        return v * limit / v_norm
    return v

def hat_to_twist(xi_hat: np.ndarray):
    """Convert se(3) matrix to 6x1 twist vector (angular | linear)."""
    w = np.array([xi_hat[2,1], xi_hat[0,2], xi_hat[1,0]])
    v = xi_hat[:3,3]
    return np.hstack((w, v))

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

class CamPoseController(Node):
    def __init__(self):
        super().__init__('cam_pose_controller')

        self.cmd_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 1)
        self.state_pub = self.create_publisher(Bool, '/homed',1)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        

        self.target_tf = TransformStamped()
        self.target_tf.header.frame_id="base_link"
        self.target_tf.header.stamp = self.get_clock().now().to_msg()
        self.target_tf.child_frame_id = "home"
        self.target_tf.transform.translation.x = 0.106
        self.target_tf.transform.translation.y = 0.265
        self.target_tf.transform.translation.z = 0.3
        self.target_tf.transform.rotation.x = 0.0
        self.target_tf.transform.rotation.y = -1.0
        self.target_tf.transform.rotation.z = 0.0
        self.target_tf.transform.rotation.w = 0.0
        
        self.error = 1e9

        self.dt = 0.01  # control loop period
        self.timer = self.create_timer(self.dt, self.timer_cb)

    def timer_cb(self):
        try:
            base_frame = "base_link"
            ee_frame = "wrist_3_link"

            if self.tf_buffer.can_transform(base_frame, ee_frame, rclpy.time.Time()):

                ee_tf = self.tf_buffer.lookup_transform(
                    base_frame, ee_frame, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.dt)
                )

                self.compute_twist(ee_tf, self.target_tf)

        except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException) as e:
            self.get_logger().warn(f"Problem getting TFs: {e}", throttle_duration_sec=5.0)

    def compute_twist(self, ee_tf: TransformStamped, target_tf: TransformStamped):
        Kp = 1

        # --- Current end-effector pose ---
        t_ee = ee_tf.transform.translation
        R_ee = quaternion_to_rotation_matrix(
            ee_tf.transform.rotation.x,
            ee_tf.transform.rotation.y,
            ee_tf.transform.rotation.z,
            ee_tf.transform.rotation.w
        )
        x_ee = np.array([t_ee.x, t_ee.y, t_ee.z])
        H_ee = np.block([[R_ee, x_ee.reshape(3,1)],
                         [np.zeros((1,3)), 1]])

        # --- Desired pose from TF ---
        t_d = target_tf.transform.translation
        R_d = quaternion_to_rotation_matrix(
            target_tf.transform.rotation.x,
            target_tf.transform.rotation.y,
            target_tf.transform.rotation.z,
            target_tf.transform.rotation.w
        )
        x_d = np.array([t_d.x, t_d.y, t_d.z])

        #x_d = np.array([0.3,0.3,0.3])
        #R_d = RPY_to_R(-np.pi/2,0,0)

        H_d = np.block([[R_d, x_d.reshape(3,1)],
                        [np.zeros((1,3)), 1]])

        # --- Compute error in se(3) ---
        err_hat = logm(np.linalg.inv(H_ee) @ H_d)
        xi_err = hat_to_twist(err_hat)

        self.error = np.linalg.norm(xi_err)

        ctrl = Kp * xi_err
        v = ctrl[3:];w = ctrl[:3]
        v = clamp(v,0.05); w = clamp(w,0.15)



        # --- Publish twist command ---
        twist_out = Twist()

        twist_out.angular.x = float(w[0])
        twist_out.angular.y = float(w[1])
        twist_out.angular.z = float(w[2])

        twist_out.linear.x = float(v[0])
        twist_out.linear.y = float(v[1])
        twist_out.linear.z = float(v[2])

        self.cmd_pub.publish(twist_out)

def main(args=None):
    rclpy.init(args=args)
    node = CamPoseController()
    
    while node.error > 1e-3:
        rclpy.spin_once(node)

    
    node.state_pub.publish(Bool(data=True))
    rclpy.spin_once(node)
    node.get_logger().warn("HOMING COMPLETE")
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
