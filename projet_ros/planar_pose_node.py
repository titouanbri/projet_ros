#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformBroadcaster
import numpy as np
import tf_transformations # Assurez-vous d'avoir sudo apt install ros-humble-tf-transformations

class PlanarPoseNode(Node):
    def __init__(self):
        super().__init__('planar_pose_node')

        self.get_logger().info("Planar Pose Node initialized (Ray-Plane Intersection Z=0)")

        # --- PARAMÈTRES DE LISSAGE ---
        self.alpha_pos = 0.6  # 1.0 = pas de lissage, 0.1 = très lent
        self.prev_pos = None  # [x, y, z]

        # --- TF SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- CAMERA INIT ---
        self.camera_matrix = None
        self.camera_frame_id = "camera_color_optical_frame" # Par défaut, sera mis à jour via camera_info
        self.camera_matrix_inv = None

        # --- SUBSCRIBERS ---
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        self.center_sub = self.create_subscription(
            Point,
            '/detected_center',
            self.center_callback,
            10
        )

        # --- PUBLISHERS ---
        self.pose_pub = self.create_publisher(PoseStamped, '/puck/pose', 10)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            K = np.array(msg.k).reshape((3, 3))
            self.camera_matrix = K
            self.camera_matrix_inv = np.linalg.inv(K)
            self.camera_frame_id = msg.header.frame_id
            self.get_logger().info(f"Caméra calibrée. Frame: {self.camera_frame_id}")

    def center_callback(self, msg):
        """
        Reçoit le centre (u, v) en pixels via geometry_msgs/Point (x=u, y=v)
        """
        if self.camera_matrix is None:
            return

        # Récupérer les coordonnées pixels
        u = msg.x
        v = msg.y

        try:
            trans = self.tf_buffer.lookup_transform(
                'base_link',
                self.camera_frame_id,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().warn(f'Impossible de récupérer la TF: {ex}')
            return

        # Calcul du rayon dans le repère CAMÉRA

        pixel_vec = np.array([u, v, 1.0])
        # Rayon normalisé dans le repère caméra (x, y, 1)

        ray_camera = self.camera_matrix_inv @ pixel_vec
        ray_camera = ray_camera / np.linalg.norm(ray_camera)

        # Transformation du rayon dans le repère BASE_LINK
        t_cam = trans.transform.translation
        r_cam = trans.transform.rotation
        
        # Position de la caméra (Origine du rayon)
        O = np.array([t_cam.x, t_cam.y, t_cam.z])

        # Quaternion vers Matrice de rotation
        q = [r_cam.x, r_cam.y, r_cam.z, r_cam.w]
        R = tf_transformations.quaternion_matrix(q)[:3, :3]

        # Vecteur direction du rayon exprimé dans la base
        D = R @ ray_camera

        # Equation paramétrique : P = O + t * D
        # P.z = 0  => O.z + t * D.z = 0  => t = - O.z / D.z
        
        if abs(D[2]) < 1e-6:
            self.get_logger().warn("Le rayon est parallèle au sol, pas d'intersection.")
            return

        t = -O[2] / D[2]

        if t < 0:
            self.get_logger().warn("L'intersection est derrière la caméra.")
            return

        # Point d'intersection (X, Y, 0)
        intersection = O + t * D

        # Lissage et Publication
        current_pos = intersection

        if self.prev_pos is None:
            self.prev_pos = current_pos
        else:
            # Filtre passe-bas
            self.prev_pos = (self.alpha_pos * current_pos) + ((1.0 - self.alpha_pos) * self.prev_pos)

        x_out, y_out, z_out = self.prev_pos

        # Création du message PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = "base_link"  

        pose_msg.pose.position.x = x_out
        pose_msg.pose.position.y = y_out
        pose_msg.pose.position.z = 0.0 # On force 0.0 car hypothèse

        # on le considère a plat
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

        # 7. TF Broadcast 
        t_pub = TransformStamped()
        t_pub.header.stamp = pose_msg.header.stamp
        t_pub.header.frame_id = "base_link"
        t_pub.child_frame_id = "puck_link"
        
        t_pub.transform.translation.x = x_out
        t_pub.transform.translation.y = y_out
        t_pub.transform.translation.z = 0.0
        t_pub.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(t_pub)

def main(args=None):
    rclpy.init(args=args)
    node = PlanarPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()