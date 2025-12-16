#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener, TransformException
import numpy as np
from scipy.spatial.transform import Rotation as R

class UR3eIVK(Node):
    """Minimal Inverse Velocity Kinematics Node for UR3e using ROS2 Humble with low-pass filter, wrt wrist_3"""
    def __init__(self):
        super().__init__("ur3e_ivk")

        # Parameters
        self.period = 0.005  # 100 Hz
        self.arm_id = "ur3e"
        self.cmd_in = Twist()
        self.current_dq = np.zeros(6)  # UR3e has 6 movable joints
        self.filtered_dq = np.zeros(6)  # for low-pass filter
        self.alpha = 0.1  # LPF coefficient (0 < alpha <= 1)

        # Link list (base → joints → EE)
        self.lookup_list = [
            "base_link_inertia",
            "shoulder_link",
            "upper_arm_link",
            "forearm_link",
            "wrist_1_link",
            "wrist_2_link",
            "wrist_3_link",
        ]
        self.link_names = self.lookup_list[1:7]  # only joints (6 for UR3e)

        # TF buffer
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(Twist, "/ee_velocity_cmd", self.cmd_in_callback, 1)
        self.create_subscription(JointState, "/joint_states", self.joint_state_callback, 1)

        # Publisher
        self.cmd_out_pub = self.create_publisher(Float64MultiArray, "/forward_velocity_controller/commands", 1)

        # Timer
        self.create_timer(self.period, self.process)

    def process(self):
        # Lookup all joint transforms from base
        links = []
        ref = self.lookup_list[-1] #self.lookup_list[0]
        try:
            for link in self.link_names:
                tf = self.tf_buffer.lookup_transform(ref, link, rclpy.time.Time())
                links.append(tf)
            # Also get wrist_3 position
            wrist3_tf = self.tf_buffer.lookup_transform(ref, "wrist_3_link", rclpy.time.Time())
            p_wrist3 = np.array([wrist3_tf.transform.translation.x,
                                 wrist3_tf.transform.translation.y,
                                 wrist3_tf.transform.translation.z])
        except TransformException:
            self.get_logger().warn("TF lookup failed; waiting for transforms...")
            return

        N = len(links)
        if N == 0:
            return

        # Positions & Z axes
        ps = [np.array([l.transform.translation.x,
                        l.transform.translation.y,
                        l.transform.translation.z]) for l in links]
        zs = []
        for l in links:
            q = l.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            zs.append(rot[:, 2])  # Z axis

        # Jacobian (linear w.r.t wrist_3)
        J_lin = np.zeros((3, N))
        J_ang = np.zeros((3, N))
        for i in range(N):
            J_lin[:, i] = np.cross(zs[i], - ps[i])
            J_ang[:, i] = zs[i]
        J = np.vstack((J_lin, J_ang))

        # Twist vector
        V = np.hstack((
            self.cmd_in.linear.x,
            self.cmd_in.linear.y,
            self.cmd_in.linear.z,
            self.cmd_in.angular.x,
            self.cmd_in.angular.y,
            self.cmd_in.angular.z
        ))

        # Damped pseudo-inverse
        damp_coeff = 1e-2
        Jt = J.T
        dq = Jt @ np.linalg.inv(J @ Jt + damp_coeff * np.eye(6)) @ V

        # Low-pass filter: filtered_dq = alpha * new + (1-alpha) * previous
        self.filtered_dq = self.alpha * dq + (1 - self.alpha) * self.filtered_dq

        # Publish filtered joint velocities
        msg = Float64MultiArray(data=self.filtered_dq.tolist())
        self.cmd_out_pub.publish(msg)

    def cmd_in_callback(self, msg: Twist):
        self.cmd_in = msg

    def joint_state_callback(self, msg: JointState):
        self.current_dq = np.array(msg.velocity[:6])  # 6 joints


def main(args=None):
    rclpy.init(args=args)
    node = UR3eIVK()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
