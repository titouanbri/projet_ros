#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

class SupervisorNode(Node):
    def __init__(self):
        super().__init__('supervisor_node')

        # --- Paramètres ---
        self.target_frame = 'puck_link'  # L'objet suivi par le PBVS
        self.base_frame = 'base_link'
        self.desired_frame_name = 'desired_ee'
        self.actual_ee_frame = 'wrist_3_link' # La vraie frame du robot à contrôler (à adapter selon votre robot)

        # --- Machine à états ---
        # States: 'PBVS', 'TRANSITION', 'POSE_CONTROL', 'FINISHED'
        self.state = 'PBVS'
        self.desired_transform = None

        # --- Publishers (Commandes) ---
        self.pub_cmd_pbvs = self.create_publisher(Bool, '/ctrl/pbvs/enable', 10)
        self.pub_cmd_pose = self.create_publisher(Bool, '/auto_pose_control_enabled', 10)

        # --- Subscribers (Feedback) ---
        self.sub_feedback_pbvs = self.create_subscription(Bool, '/ctrl/pbvs/done', self.pbvs_done_cb, 10)
        self.sub_feedback_pose = self.create_subscription(Bool, '/ctrl/pose/done', self.pose_done_cb, 10)
        
        # Flags de feedback
        self.pbvs_finished = False
        self.pose_finished = False

        # --- TF Tools ---
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Timer principal (10Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        self.get_logger().info("Supervisor Node Started")

    def pbvs_done_cb(self, msg):
        if msg.data:
            self.pbvs_finished = True

    def pose_done_cb(self, msg):
        if msg.data:
            self.pose_finished = True

    def control_loop(self):
        # Envoi des commandes selon l'état
        cmd_pbvs = Bool()
        cmd_pose = Bool()

        if self.state == 'PBVS':
            cmd_pbvs.data = True
            cmd_pose.data = False
            
            if self.pbvs_finished:
                self.get_logger().info("PBVS terminé. Transition vers Pose Control.")
                self.state = 'TRANSITION'

        elif self.state == 'TRANSITION':
            # On fige la 'desired_ee' basée sur la position actuelle du puck
            # mais avec une orientation imposée (ex: pince vers le bas)
            success = self.define_desired_pose()
            if success:
                self.state = 'POSE_CONTROL'
            else:
                self.get_logger().warn("Attente TF pour définir desired_ee...")

        elif self.state == 'POSE_CONTROL':
            cmd_pbvs.data = False
            cmd_pose.data = True
            
            # On doit continuer à publier la TF desired_ee
            if self.desired_transform:
                self.desired_transform.header.stamp = self.get_clock().now().to_msg()
                self.tf_broadcaster.sendTransform(self.desired_transform)

            if self.pose_finished:
                self.get_logger().info("Pose Control terminé. Mission accomplie.")
                self.state = 'FINISHED'

        elif self.state == 'FINISHED':
            cmd_pbvs.data = False
            cmd_pose.data = False
            # Optionnel : continuer à publier desired_ee pour maintenir la position
            if self.desired_transform:
                self.desired_transform.header.stamp = self.get_clock().now().to_msg()
                self.tf_broadcaster.sendTransform(self.desired_transform)

        # Publication des commandes
        self.pub_cmd_pbvs.publish(cmd_pbvs)
        self.pub_cmd_pose.publish(cmd_pose)
        print(f"State: {self.state}, PBVS_cmd: {cmd_pbvs.data}, Pose_cmd: {cmd_pose.data}")

    def define_desired_pose(self):
        try:
            # On cherche où est le puck par rapport à la base
            t = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.target_frame,
                rclpy.time.Time()
            )


            # On récupère l'ORIENTATION actuelle du robot (pour la figer)
            t_current_ee = self.tf_buffer.lookup_transform(
                self.base_frame,
                self.actual_ee_frame, # La frame réelle du robot
                rclpy.time.Time()
            )
            
            # Création du message TransformStamped pour 'desired_ee'
            d = TransformStamped()
            d.header.stamp = self.get_clock().now().to_msg()
            d.header.frame_id = self.base_frame 
            # d.header.frame_id = 'puck_link' # On peut aussi faire desired_ee fixé au puck, ça rend la tâche plus facile pour le pose control

        

            d.child_frame_id = self.desired_frame_name
            
            # Position : On garde la position du puck (ou on ajoute un offset Z pour l'approche)
            d.transform.translation.x = t.transform.translation.x #0
            d.transform.translation.y = t.transform.translation.y #0
            d.transform.translation.z = t.transform.translation.z + 0.15 # + 0.15 par exemple

            d.transform.rotation.x = t_current_ee.transform.rotation.x #1
            d.transform.rotation.y = t_current_ee.transform.rotation.y #0
            d.transform.rotation.z = t_current_ee.transform.rotation.z #0
            d.transform.rotation.w = t_current_ee.transform.rotation.w #0

            # d.transform.translation.x = 0.
            # d.transform.translation.y = 0.
            # d.transform.translation.z = 0.15
            # d.transform.rotation.x = 1.
            # d.transform.rotation.y = 0.
            # d.transform.rotation.z = 0.
            # d.transform.rotation.w = 0.

            
            self.desired_transform = d
            return True
            
        except Exception as e:
            self.get_logger().error(f"Erreur TF: {e}")
            return False

def main(args=None):
    rclpy.init(args=args)
    node = SupervisorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()