import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist, TransformStamped
import tf2_ros
import numpy as np
#import tf_transformations as tft
from scipy.linalg import expm, logm

def unskew(S):
    w = np.array([S[2,1],S[0,2],S[1,0]]).reshape(3,1)
    return w

def hat_to_twist(xi_hat : np.ndarray):
    R = xi_hat[:3,:3]
    t = xi_hat[:3,3]
    w = unskew(R)
    xi = np.concatenate((w.flatten(),t.flatten())).reshape(6,)
    return xi

def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    # Normalize quaternion
    norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Compute rotation matrix elements
    r00 = 1 - 2*(qy**2 + qz**2)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx**2 + qz**2)
    r12 = 2*(qy*qz - qx*qw)
    
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx**2 + qy**2)
    
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])


def skew(vector):
    """Return the skew-symmetric matrix of a 3-vector"""
    v = np.asarray(vector).copy().reshape(3)
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def adj(H : np.ndarray):
    R = H[:3,:3]
    t = H[:3,3]
    H_bar = np.block([
        [R, np.zeros((3,3))],
        [skew(t)@R, R]])
    return H_bar 



class CamPoseController(Node):
    def __init__(self):
        super().__init__('cam_pose_controller')

        self.cam_tf: TransformStamped = None
        self.marker_tf: TransformStamped = None
        self.ee_tf: TransformStamped = None

        self.dt = 1e-2

        self.cmd_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(self.dt, self.timer_cb)

    def timer_cb(self):
        try:
            base = "base_link"
            camera = "camera_link"
            marker = "aruco"
            ee = "wrist_3_link"

            if self.tf_buffer.can_transform(base, camera, rclpy.time.Time()):
                self.cam_tf = self.tf_buffer.lookup_transform(
                    base,
                    camera,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.dt),
                )

            if self.tf_buffer.can_transform(base, marker, rclpy.time.Time()):
                self.marker_tf = self.tf_buffer.lookup_transform(
                    base,
                    marker,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.dt),
                )

            if self.tf_buffer.can_transform(base, ee, rclpy.time.Time()):
                self.ee_tf = self.tf_buffer.lookup_transform(
                    base,
                    ee,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=self.dt),
                )

            self.compute_twist()

        except (
            tf2_ros.LookupException,
            tf2_ros.ExtrapolationException,
            tf2_ros.ConnectivityException,
        ) as e:
            self.get_logger().warn(
                f"Problem getting TFs : {e}", throttle_duration_sec=5.0
            )
        except Exception as e:
            self.get_logger().warn(
                f"Unhandled exception at get_tfs: {e}", throttle_duration_sec=5.0
            )

    def compute_twist(self):
        try:
            if self.marker_tf is not None and self.cam_tf is not None and self.ee_tf is not None:
                Kp = 0.1

                #----------Calculating cam pose wrt base_link--------------------
                t_cam = self.cam_tf.transform.translation
                x_cam = np.array([t_cam.x, t_cam.y, t_cam.z])
                q = self.cam_tf.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                R_cam = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)[:3, :3]

                H_c = np.block([[R_cam, x_cam.reshape(3,1)],
                                [np.zeros((1,3)), 1]]) #wrt world
                #----------Calculating cam pose wrt base_link--------------------

                #----------Calculating ARUCO marker pose wrt base_link--------------------
                t_marker = self.marker_tf.transform.translation
                x_marker = np.array([t_marker.x, t_marker.y, t_marker.z])
                q = self.marker_tf.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                R_marker = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)[:3, :3]

                H_aruco_wrt_world = np.block([[R_marker, x_marker.reshape(3,1)],
                                [np.zeros((1,3)), 1]])
                #----------Calculating ARUCO marker pose wrt base_link--------------------

                #----------Calculating EE pose wrt base_link--------------------
                q = self.ee_tf.transform.rotation
                quat = [q.x, q.y, q.z, q.w]
                R_ee = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)[:3, :3]
                t_ee = self.ee_tf.transform.translation

                x_ee = np.array([t_ee.x, t_ee.y, t_ee.z])

                H_ee = np.block([[R_ee, x_ee.reshape(3,1)],
                                [np.zeros((1,3)), 1]])
                
                #----------Calculating EE pose wrt base_link--------------------
                

                
                Hd_wrt_aruco = np.eye(4)
                Hd_wrt_aruco[:3,3] = np.array([0,0,0.03])

                
                
                H_d_wrt_world = H_aruco_wrt_world @ Hd_wrt_aruco

                err_hat = logm(np.linalg.inv(H_c) @ H_d_wrt_world)
                xi_err = hat_to_twist(err_hat) #wrt current cam pose

                
                


                ctrl = Kp * adj(H_ee) @ xi_err



                twist_out = Twist()
                

                twist_out.linear.x = float(ctrl[3])
                twist_out.linear.y = float(ctrl[4])
                twist_out.linear.z = float(ctrl[5])
                twist_out.angular.x = float(ctrl[0]) 
                twist_out.angular.y = float(ctrl[1]) 
                twist_out.angular.z = float(ctrl[2]) 


                self.cmd_pub.publish(twist_out)
            else:
                self.get_logger().info(
                    "TFs not initialized yet", throttle_duration_sec=5.0
                )

        except Exception as e:
            self.get_logger().warn(
                f"Error at compute_twist: {e}", throttle_duration_sec=5.0
            )


def main(args=None):
    rclpy.init(args=args)
    node = CamPoseController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
