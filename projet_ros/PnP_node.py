#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo
from geometry_msgs.msg import Polygon, PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import cv2
import numpy as np
import math

class PnPNode(Node):
    def __init__(self):
        super().__init__('pnp_node')

        self.get_logger().info("PnP Node initialized (Puck 3D Pose + Vertical Alignment)")

        # --- TF BUFFER & LISTENER ---
        # Nécessaire pour connaître l'orientation de la caméra par rapport à base_link
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # --- PARAMÈTRES DE LISSAGE (SOLUTION C) ---
        self.alpha_pos = 0.6   # Filtrage très fort pour la position
        self.alpha_rot = 0.2   # Filtrage pour l'orientation

        # Variables pour stocker l'état précédent
        self.prev_pos = None    # [x, y, z]
        self.prev_quat = None   # [x, y, z, w]

        # --- CONFIGURATION PnP ---
        self.target_width = 0.031
        self.target_height = 0.031

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

    def matrix_to_quaternion(self, R):
        """Convertit une matrice de rotation 3x3 en quaternion [x, y, z, w]."""
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
            
        return np.array(q)

    def quat_to_mat(self, q):
        """Convertit un quaternion (de geometry_msgs ou objet avec .x .y .z .w) en matrice 3x3."""
        x, y, z, w = q.x, q.y, q.z, q.w
        return np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ], dtype=np.float64)

    def compute_constrained_orientation(self, rvec, z_target_cam):
        """
        Calcule une orientation où l'axe Z de l'objet est aligné avec z_target_cam,
        mais l'axe X (Yaw) respecte la détection du PnP.
        """
        # 1. Conversion PnP Rodrigues -> Matrice
        R_pnp, _ = cv2.Rodrigues(rvec)
        
        # 2. Correction initiale (comme dans votre code original pour aligner les axes PnP)
        rot_x_180 = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ], dtype=np.float64)
        R_pnp = np.dot(R_pnp, rot_x_180)

        # 3. Extraction de l'axe X détecté (C'est la direction "avant" du puck)
        x_pnp = R_pnp[:, 0]

        # 4. Construction de la nouvelle base
        # Z_new : On force l'axe Z à être le vecteur cible (le ciel vu depuis la caméra)
        z_new = z_target_cam
        norm_z = np.linalg.norm(z_new)
        if norm_z > 1e-6:
            z_new /= norm_z
        else:
            z_new = np.array([0, 0, -1]) # Fallback

        # X_new : On projette x_pnp sur le plan orthogonal à z_new
        # X_proj = X - (X . Z) * Z
        dot = np.dot(x_pnp, z_new)
        x_new = x_pnp - (dot * z_new)
        
        norm_x = np.linalg.norm(x_new)
        if norm_x > 1e-6:
            x_new /= norm_x
        else:
            # Cas rare : X est parallèle à Z (Gimbal lock), on choisit un X arbitraire
            x_new = np.cross(np.array([0, 1, 0]), z_new)
            if np.linalg.norm(x_new) < 1e-6:
                x_new = np.cross(np.array([1, 0, 0]), z_new)
            x_new /= np.linalg.norm(x_new)

        # Y_new : Produit vectoriel pour finir la base orthonormée
        y_new = np.cross(z_new, x_new)

        # 5. Construction de la matrice finale
        R_final = np.column_stack((x_new, y_new, z_new))

        return self.matrix_to_quaternion(R_final)

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
            # --- RECUPERATION DU VECTEUR "CIEL" (Z WORLD) DANS LE REPERE CAMERA ---
            z_target_cam = np.array([0.0, 0.0, -1.0]) # Valeur par défaut (face caméra)
            
            try:
                # On cherche la transfo : Base (World) -> Camera
                # Cela nous permet de savoir comment le vecteur Z=(0,0,1) de la base est vu par la caméra
                t_stamped = self.tf_buffer.lookup_transform(
                    'camera_color_optical_frame', # Target frame
                    'base_link',                  # Source frame (Repère Global)
                    rclpy.time.Time()
                )
                
                # Convertir le quaternion de la transfo en matrice de rotation
                R_base_to_cam = self.quat_to_mat(t_stamped.transform.rotation)
                
                # Le vecteur Z du monde est [0, 0, 1] dans base_link.
                # Une fois tourné dans le repère caméra, c'est simplement la 3ème colonne de la matrice.
                z_target_cam = R_base_to_cam @ np.array([0, 0, 1])

            except Exception as e:
                # Si TF pas prêt, on garde le comportement par défaut (mais on log pas à chaque frame pour éviter le spam)
                pass

            # 1. Récupération des valeurs
            raw_pos = np.array([tvec[0][0], tvec[1][0], tvec[2][0]])
            
            # Calcul de l'orientation contrainte (Z aligné sur base_link Z, X aligné sur détection PnP)
            raw_quat = self.compute_constrained_orientation(rvec, z_target_cam)

            # 2. Application du filtre (Low Pass Filter)
            if self.prev_pos is None:
                self.prev_pos = raw_pos
                self.prev_quat = raw_quat
                smoothed_pos = raw_pos
                smoothed_quat = raw_quat
            else:
                smoothed_pos = (self.alpha_pos * raw_pos) + ((1.0 - self.alpha_pos) * self.prev_pos)
                smoothed_quat = (self.alpha_rot * raw_quat) + ((1.0 - self.alpha_rot) * self.prev_quat)
                
                norm = np.linalg.norm(smoothed_quat)
                if norm > 0:
                    smoothed_quat /= norm
                
                self.prev_pos = smoothed_pos
                self.prev_quat = smoothed_quat

            # 3. Préparation des messages
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