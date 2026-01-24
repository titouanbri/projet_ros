#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Polygon, PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import cv2
import numpy as np
import math

class PnPNode(Node):
    def __init__(self):
        super().__init__('pnp_node')

        self.get_logger().info("PnP Node initialized (Puck 3D Pose)")

        # --- PARAMÈTRES OBJET ---
        self.target_width = 0.054
        self.target_height = 0.054

        # --- PARAMÈTRES CAMÉRA PAR DÉFAUT ---
        # Si aucune info caméra n'est reçue, on utilise ces valeurs.
        # Exemple pour une résolution 640x480 avec un FOV standard (~60°)
        # fx ~ width, fy ~ width, cx = width/2, cy = height/2
        
        # Largeur/Hauteur supposées de l'image (à ajuster si vous utilisez du 1280x720, etc.)
        img_w = 640.0
        img_h = 480.0
        fx = img_w  # approximation focale
        fy = img_h
        cx = img_w / 2.0
        cy = img_h / 2.0

        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        # Pas de distorsion par défaut
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        self.get_logger().warn(f"Calibration par défaut chargée (Intrinsics: {fx}x{fy}). En attente de /camera_info pour affiner...")

        # --- INIT TF BROADCASTER ---
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- SUBSCRIBERS ---
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera_info', # Vérifiez que ce topic est correct
            self.info_callback,
            10
        )

        self.corners_sub = self.create_subscription(
            Polygon,
            '/detected_corners',
            self.corners_callback,
            10
        )

        # --- PUBLISHERS ---
        self.pose_pub = self.create_publisher(PoseStamped, '/puck/pose', 10)

    def info_callback(self, msg):
        # On ne met à jour que si on n'a pas encore reçu de vraie calibration
        # ou si vous voulez permettre la mise à jour en continu, retirez la condition 'if not self...'
        if np.linalg.norm(np.array(msg.k).reshape((3, 3))) > 0.1 :
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            
            self.get_logger().info("Calibration RÉELLE reçue via /camera_info ! Remplacement des valeurs par défaut.")
            print("Camera Matrix:\n", self.camera_matrix)
            print("Distortion Coefficients:\n", self.dist_coeffs)

    def rvec_to_quaternion(self, rvec):
        """
        Convertit un vecteur de rotation (Rodrigues) en Quaternion ROS [x, y, z, w]
        """
        R, _ = cv2.Rodrigues(rvec)
        
        tr = np.trace(R)
        q = [0, 0, 0, 0]

        if tr > 0:
            S = math.sqrt(tr + 1.0) * 2
            q[3] = 0.25 * S
            q[0] = (R[2, 1] - R[1, 2]) / S
            q[1] = (R[0, 2] - R[2, 0]) / S
            q[2] = (R[1, 0] - R[0, 1]) / S
        elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[3] = (R[2, 1] - R[1, 2]) / S
            q[0] = 0.25 * S
            q[1] = (R[0, 1] + R[1, 0]) / S
            q[2] = (R[0, 2] + R[2, 0]) / S
        elif R[1, 1] > R[2, 2]:
            S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[3] = (R[0, 2] - R[2, 0]) / S
            q[0] = (R[0, 1] + R[1, 0]) / S
            q[1] = 0.25 * S
            q[2] = (R[1, 2] + R[2, 1]) / S
        else:
            S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[3] = (R[1, 0] - R[0, 1]) / S
            q[0] = (R[0, 2] + R[2, 0]) / S
            q[1] = (R[1, 2] + R[2, 1]) / S
            q[2] = 0.25 * S
            
        return q

    def corners_callback(self, msg):
        # NOTE: On a supprimé le check "if self.camera_matrix is None" car on a des valeurs par défaut.

        # 1. Extraction des points 2D
        if len(msg.points) != 4:
            return

        image_points = np.array([
            [msg.points[0].x, msg.points[0].y], # TL
            [msg.points[1].x, msg.points[1].y], # TR
            [msg.points[2].x, msg.points[2].y], # BR
            [msg.points[3].x, msg.points[3].y]  # BL
        ], dtype=np.float32)

        # 2. Définition des points 3D de l'objet
        w = self.target_width
        h = self.target_height
        
        object_points = np.array([
            [-w / 2.0, -h / 2.0, 0.0],
            [ w / 2.0, -h / 2.0, 0.0],
            [ w / 2.0,  h / 2.0, 0.0],
            [-w / 2.0,  h / 2.0, 0.0]
        ], dtype=np.float32)

        # 3. Résolution PnP
        # Si on utilise les valeurs par défaut, la précision Z (profondeur) sera approximative
        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            x_trans = tvec[0][0]
            y_trans = tvec[1][0]
            z_trans = tvec[2][0]

            q = self.rvec_to_quaternion(rvec)

            # --- 1. Publication PoseStamped ---
            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            # pose_msg.header.frame_id = "camera_color_optical_frame"
            pose_msg.header.frame_id = "camera_link"

            
            pose_msg.pose.position.x = x_trans
            pose_msg.pose.position.y = y_trans
            pose_msg.pose.position.z = z_trans
            
            pose_msg.pose.orientation.x = q[0]
            pose_msg.pose.orientation.y = q[1]
            pose_msg.pose.orientation.z = q[2]
            pose_msg.pose.orientation.w = q[3]

            self.pose_pub.publish(pose_msg)

            # --- 2. Publication TF ---
            t = TransformStamped()
            t.header.stamp = pose_msg.header.stamp
            t.header.frame_id = pose_msg.header.frame_id
            t.child_frame_id = 'puck_link'

            t.transform.translation.x = x_trans
            t.transform.translation.y = y_trans
            t.transform.translation.z = z_trans

            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = PnPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()