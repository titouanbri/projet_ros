#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
# Ajout de Polygon et Point32 pour les coins
from geometry_msgs.msg import Point, Polygon, Point32, PoseStamped
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        self.get_logger().info("ArUco Tracking 3D + Corners")

        # --- ARUCO SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # --- PARAMÈTRES 3D ---
        self.marker_length = 0.035  # Taille en mètres
        self.target_id = None 

        self.camera_matrix = None
        self.dist_coeffs = None
        
        # --- ROS ---
        self.br = CvBridge()
        
        # 1. Abonnement Image & Info
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',  
            self.image_callback,
            10
        )
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info',
            self.info_callback,
            10
        )

        # Publishers
        self.image_pub = self.create_publisher(Image, '/aruco/result_image', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/aruco/pose', 10)
        self.center_pub = self.create_publisher(Point, '/aruco/center_pixels', 10)

        # --- NOUVEAU PUBLISHER POUR LES 4 ANGLES ---
        # On utilise Polygon car c'est une liste de points
        self.corners_pub = self.create_publisher(Polygon, '/aruco/corners_pixels', 10)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Calibration caméra reçue !")

    def image_callback(self, msg):
        if self.camera_matrix is None:
            return

        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )

        if ids is not None and len(ids) > 0:
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, 
                self.marker_length, 
                self.camera_matrix, 
                self.dist_coeffs
            )

            ids = ids.flatten()
            
            for i, marker_id in enumerate(ids):
                if self.target_id is not None and marker_id != self.target_id:
                    continue
                
                # 1. Traitement Pose 3D (existant)
                x = tvecs[i][0][0]
                y = tvecs[i][0][1]
                z = tvecs[i][0][2]

                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose.position.x = x
                pose_msg.pose.position.y = y
                pose_msg.pose.position.z = z
                self.pose_pub.publish(pose_msg)
                
                # 2. Traitement Centre Pixel (existant)
                c = corners[i][0] # c est un array de forme (4, 2) contenant les 4 coins
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                
                pt_msg = Point()
                pt_msg.x = float(cx)
                pt_msg.y = float(cy)
                self.center_pub.publish(pt_msg)

                # --- 3. NOUVEAU : PUBLICATION DES 4 ANGLES ---
                # c contient: [TopLeft, TopRight, BottomRight, BottomLeft]
                poly_msg = Polygon()
                
                for corner in c:
                    p32 = Point32()
                    p32.x = float(corner[0]) # Pixel X
                    p32.y = float(corner[1]) # Pixel Y
                    p32.z = 0.0              # Z est 0 car on est en pixels 2D
                    poly_msg.points.append(p32)
                
                self.corners_pub.publish(poly_msg)

                # --- VISUALISATION ---
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)
                
                # Affichage debug des coins sur l'image (cercles rouges)
                for corner in c:
                    cv2.circle(cv_image, (int(corner[0]), int(corner[1])), 4, (0, 0, 255), -1)

            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

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