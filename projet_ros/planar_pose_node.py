#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
import tf2_geometry_msgs
import numpy as np
import cv2
import math

class PlanarPoseNode(Node):
    def __init__(self):
        super().__init__('planar_pose_node')
        self.get_logger().info("Planar Pose Node Initialized (Z=0 assumption)")

        # --- PARAMÈTRES ---
        self.target_frame = 'base_link'  # Le repère où le plan est défini par Z=0
        self.camera_frame = None         # Sera détecté via CameraInfo
        
        # Stockage intrinsèques caméra
        self.camera_matrix = None
        self.dist_coeffs = None

        # --- TF SETUP ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- SUBSCRIPTIONS ---
        # 1. Info Caméra (pour matrice K et distorsion)
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info', 
            self.info_callback,
            10
        )

        # 2. Centre en pixels (sortie de puck_detector)
        self.pixel_sub = self.create_subscription(
            Point,
            '/detected_center',
            self.pixel_callback,
            10
        )

        # --- PUBLISHERS ---
        # Sortie équivalente à /aruco/pose
        self.pose_pub = self.create_publisher(PoseStamped, '/puck/pose', 10)

    def info_callback(self, msg):
        """Récupère les intrinsèques de la caméra une fois."""
        if self.camera_matrix is None:
            self.camera_frame = msg.header.frame_id
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info(f"Caméra calibrée. Frame: {self.camera_frame}")

    def pixel_callback(self, msg):
        """
        Reçoit le centre (u, v) en pixels, calcule la pose 3D sur le plan Z=0
        et publie PoseStamped + TF.
        """
        if self.camera_matrix is None or self.camera_frame is None:
            self.get_logger().warn("En attente des infos caméra...", throttle_duration_sec=2)
            return

        # 1. Récupérer le point pixel (u, v)
        u, v = msg.x, msg.y

        # 2. Créer le rayon (Ray Casting) dans le repère CAMÉRA
        # On dé-distord le point pour avoir des coordonnées normalisées
        uv_point = np.array([[[u, v]]], dtype=np.float32)
        norm_point = cv2.undistortPoints(uv_point, self.camera_matrix, self.dist_coeffs)
        x_norm = norm_point[0, 0, 0]
        y_norm = norm_point[0, 0, 1]

        # Vecteur rayon dans le repère caméra : O_cam(0,0,0) -> P_img(x_n, y_n, 1)
        ray_direction_cam = np.array([x_norm, y_norm, 1.0])

        # 3. Récupérer la transform Caméra -> Base Link (TF)
        try:
            # On cherche la transfo la plus récente
            trans = self.tf_buffer.lookup_transform(
                self.target_frame,      # Target: base_link
                self.camera_frame,      # Source: camera_color_optical_frame
                rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"Impossible de récupérer la TF {self.target_frame} -> {self.camera_frame}: {e}")
            return

        # 4. Transformer l'origine et le rayon dans le repère BASE_LINK
        
        # Position de la caméra dans base_link (C'est l'origine du rayon)
        cam_origin = np.array([
            trans.transform.translation.x,
            trans.transform.translation.y,
            trans.transform.translation.z
        ])

        # Rotation (Quaternion -> Matrice) pour tourner le vecteur rayon
        q = trans.transform.rotation
        import transforms3d # Si disponible, sinon on fait manuellement ou via scipy
        # Méthode simple via scipy si disponible, ou manuelle ici pour limiter les dépendances :
        # Conversion Quaternion [x, y, z, w] -> Matrice de rotation
        R = self.quat_to_mat([q.x, q.y, q.z, q.w])
        
        # Rayon exprimé dans base_link
        ray_direction_base = R @ ray_direction_cam

        # 5. Calcul d'intersection avec le plan Z = 0
        # Equation paramétrique : P = Origin + t * Direction
        # On veut P.z = 0  =>  Origin.z + t * Direction.z = 0
        # t = - Origin.z / Direction.z

        if abs(ray_direction_base[2]) < 1e-6:
            self.get_logger().warn("Le rayon est parallèle au plan Z=0, pas d'intersection.")
            return

        t = -cam_origin[2] / ray_direction_base[2]

        if t < 0:
            self.get_logger().warn("L'intersection est derrière la caméra.")
            return

        # Coordonnées du point d'intersection
        intersection = cam_origin + t * ray_direction_base

        # 6. Publication de la PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = self.target_frame # base_link

        pose_msg.pose.position.x = intersection[0]
        pose_msg.pose.position.y = intersection[1]
        pose_msg.pose.position.z = 0.0 # Par définition

        # Orientation : On ne peut pas connaître le lacet (Yaw) juste avec un point.
        # On met l'identité (aligné avec base_link) ou on garde Z vers le haut.
        pose_msg.pose.orientation.x = 0.0
        pose_msg.pose.orientation.y = 0.0
        pose_msg.pose.orientation.z = 0.0
        pose_msg.pose.orientation.w = 1.0

        self.pose_pub.publish(pose_msg)

        # 7. Diffusion du TF (Transform)
        # Similaire à aruco qui publie "aruco_id"
        t_msg = TransformStamped()
        t_msg.header.stamp = pose_msg.header.stamp
        t_msg.header.frame_id = self.target_frame
        t_msg.child_frame_id = "puck_frame"

        t_msg.transform.translation.x = intersection[0]
        t_msg.transform.translation.y = intersection[1]
        t_msg.transform.translation.z = 0.0
        t_msg.transform.rotation = pose_msg.pose.orientation

        self.tf_broadcaster.sendTransform(t_msg)

    def quat_to_mat(self, q):
        """Convertit un quaternion [x, y, z, w] en matrice de rotation 3x3."""
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
            [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])

def main(args=None):
    rclpy.init(args=args)
    node = PlanarPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()