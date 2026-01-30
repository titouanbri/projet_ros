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

        self.get_logger().info("PnP Node initialized (Puck 3D Pose + Strong Filtering)")

        # --- PARAMÈTRES DE LISSAGE (SOLUTION C) ---
        # Alpha détermine la réactivité : 
        # 1.0 = pas de filtre (réactif mais bruité)
        # 0.1 = très lissé (stable mais lent à converger)
        # Puisque l'objet est immobile, on peut mettre une valeur très basse.
        self.alpha_pos = 0.6   # Filtrage très fort pour la position
        self.alpha_rot = 0.2    # Filtrage pour l'orientation

        # Variables pour stocker l'état précédent
        self.prev_pos = None    # [x, y, z]
        self.prev_quat = None   # [x, y, z, w]

        # --- CONFIGURATION PnP ---
        self.target_width = 0.025
        self.target_height = 0.025

        # Parametres cam par défaut
        self.not_get = True
        img_w = 640.0
        img_h = 480.0
        fx = img_w  
        fy = img_h
        cx = img_w / 2.0
        cy = img_h / 2.0

        self.camera_matrix = np.array([
            [fx,  0, cx],
            [ 0, fy, cy],
            [ 0,  0,  1]
        ], dtype=np.float64)

        self.dist_coeffs = np.zeros((5, 1), dtype=np.float64)
        
        self.get_logger().warn(f"Calibration par défaut chargée. En attente de /camera_info...")

        self.tf_broadcaster = TransformBroadcaster(self)

        # Subscribers
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        self.corners_sub = self.create_subscription(
            Polygon,
            '/detected_corners',
            self.corners_callback,
            10
        )

        # Publishers
        self.pose_pub = self.create_publisher(PoseStamped, '/puck/pose', 10)

    def info_callback(self, msg):
        if np.linalg.norm(np.array(msg.k).reshape((3, 3))) > 0.1 and self.not_get:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.not_get = False
            self.get_logger().info("Calibration RÉELLE reçue !")

    def rvec_to_quaternion(self, rvec):
        # Conversion Rodrigues
        R, _ = cv2.Rodrigues(rvec)
        
        # Rotation de 180 degrés autour de l'axe X local pour corriger l'orientation
        rot_x_180 = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float64)

        R = np.dot(R, rot_x_180)

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
            
        return np.array(q) # Retourne un numpy array pour faciliter les calculs

    def corners_callback(self, msg):
        if len(msg.points) != 4:
            return

        image_points = np.array([
            [msg.points[0].x, msg.points[0].y], 
            [msg.points[1].x, msg.points[1].y], 
            [msg.points[2].x, msg.points[2].y], 
            [msg.points[3].x, msg.points[3].y] 
        ], dtype=np.float32)

        w = self.target_width
        h = self.target_height
        
        object_points = np.array([
            [-w / 2.0, -h / 2.0, 0.0],
            [ w / 2.0, -h / 2.0, 0.0],
            [ w / 2.0,  h / 2.0, 0.0],
            [-w / 2.0,  h / 2.0, 0.0]
        ], dtype=np.float32)

        success, rvec, tvec = cv2.solvePnP(
            object_points, 
            image_points, 
            self.camera_matrix, 
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if success:
            # 1. Récupération des valeurs brutes
            raw_pos = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
            raw_quat = self.rvec_to_quaternion(rvec) # [x, y, z, w]

            # 2. Application du filtre (Low Pass Filter)
            if self.prev_pos is None:
                # Initialisation
                self.prev_pos = raw_pos
                self.prev_quat = raw_quat
                smoothed_pos = raw_pos
                smoothed_quat = raw_quat
            else:
                # Filtrage Position
                smoothed_pos = (self.alpha_pos * raw_pos) + ((1.0 - self.alpha_pos) * self.prev_pos)
                
                # Filtrage Orientation (LERP simple + Normalisation)
                # Note: Pour de très petits changements, LERP est suffisant. SLERP est mieux mais plus coûteux.
                smoothed_quat = (self.alpha_rot * raw_quat) + ((1.0 - self.alpha_rot) * self.prev_quat)
                # Renormalisation obligatoire du quaternion
                norm = np.linalg.norm(smoothed_quat)
                if norm > 0:
                    smoothed_quat /= norm
                
                # Mise à jour de l'état précédent
                self.prev_pos = smoothed_pos
                self.prev_quat = smoothed_quat

            # 3. Préparation des messages avec les valeurs LISSÉES
            x_out, y_out, z_out = smoothed_pos
            qx_out, qy_out, qz_out, qw_out = smoothed_quat

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "camera_color_optical_frame"
            
            pose_msg.pose.position.x = x_out
            pose_msg.pose.position.y = y_out
            pose_msg.pose.position.z = z_out
            
            pose_msg.pose.orientation.x = qx_out
            pose_msg.pose.orientation.y = qy_out
            pose_msg.pose.orientation.z = qz_out
            pose_msg.pose.orientation.w = qw_out

            self.pose_pub.publish(pose_msg)

            # 4. TF Broadcast
            t = TransformStamped()
            t.header.stamp = pose_msg.header.stamp
            t.header.frame_id = pose_msg.header.frame_id
            t.child_frame_id = 'puck_link'

            t.transform.translation.x = x_out
            t.transform.translation.y = y_out
            t.transform.translation.z = z_out

            t.transform.rotation.x = qx_out
            t.transform.rotation.y = qy_out
            t.transform.rotation.z = qz_out
            t.transform.rotation.w = qw_out

            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = PnPNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()