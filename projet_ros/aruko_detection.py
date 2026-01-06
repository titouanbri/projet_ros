#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Polygon, Point32, PoseStamped
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        self.get_logger().info("ArUco Tracking 3D (Pose Estimation)")

        # --- ARUCO SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # --- PARAMÈTRES 3D (CRUCIAL) ---
        self.marker_length = 0.035  # <--- TAILLE RÉELLE DU MARQUEUR EN MÈTRES (ici 5cm)
        self.target_id = None 

        # Stockage de la calibration caméra
        # self.camera_matrix = np.array([
        #     [600.0, 0.0,   320.0],
        #     [0.0,   600.0, 240.0],
        #     [0.0,   0.0,   1.0]
        # ], dtype=np.float32)

        # self.dist_coeffs = np.zeros((5, 1), dtype=np.float32) # Pas de distorsion (supposé)

        # self.get_logger().info("Calibration 'Fake' chargée manuellement !")

        self.camera_matrix = None
        self.dist_coeffs = None
        # --- ROS ---
        self.br = CvBridge()
        
        # 1. Abonnement à l'image
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  
            self.image_callback,
            10
        )

        # 2. Abonnement aux infos caméra (pour la calibration)
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/aruco/result_image', 10)
        # On publie maintenant une PoseStamped (Position + Orientation 3D)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        
        # Garde les anciens publishers pixel au cas où
        self.center_pub = self.create_publisher(Point, '/aruco/center_pixels', 10)

    def info_callback(self, msg):
        """ Récupère la matrice intrinsèque K et la distorsion D une seule fois """
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Calibration caméra reçue !")

    def image_callback(self, msg):
        # On ne peut pas calculer la 3D sans calibration
        if self.camera_matrix is None:
            self.get_logger().warn("En attente de /camera_info...", throttle_duration_sec=2)
            return

        # ROS → OpenCV
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # --- Détection ArUco ---
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0:
            # --- ESTIMATION DE POSE 3D ---
            # rvecs: vecteur de rotation, tvecs: vecteur de translation (X,Y,Z)
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 
                self.marker_length, 
                self.camera_matrix, 
                self.dist_coeffs
            )

            ids = ids.flatten()
            
            for i, marker_id in enumerate(ids):
                # Filtrage ID
                if self.target_id is not None and marker_id != self.target_id:
                    continue
                
                # Récupération de la translation (X, Y, Z) en mètres
                # tvecs[i] est de forme (1, 3) -> [x, y, z]
                x = tvecs[i][0][0]
                y = tvecs[i][0][1]
                z = tvecs[i][0][2]

                # --- Publication ROS (Pose 3D) ---
                pose_msg = PoseStamped()
                pose_msg.header = msg.header # Garder le timestamp de l'image
                pose_msg.pose.position.x = x
                pose_msg.pose.position.y = y
                pose_msg.pose.position.z = z
                # (Optionnel) Convertir rvec en quaternion pour l'orientation
                # Ici on laisse l'orientation à 0 pour simplifier l'exemple
                
                self.pose_pub.publish(pose_msg)
                
                # --- Publication Pixel (Legacy) ---
                # Calcul du centre 2D pour affichage
                c = corners[i][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                pt_msg = Point()
                pt_msg.x = float(cx)
                pt_msg.y = float(cy)
                self.center_pub.publish(pt_msg)

                # --- VISUALISATION ---
                # Dessine le repère 3D (axe X:rouge, Y:vert, Z:bleu) sur l'image
                cv2.drawFrameAxes(
                    cv_image, 
                    self.camera_matrix, 
                    self.dist_coeffs, 
                    rvecs[i], 
                    tvecs[i], 
                    0.03 # Longueur des axes visuels (3cm)
                )
                
                # Afficher les coordonnées Z à l'écran
                cv2.putText(cv_image, f"Z: {z:.2f}m", (int(cx), int(cy)+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Dessine les carrés autour des markers
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        # Publication image debug
        out_msg = self.br.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoTrackingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()