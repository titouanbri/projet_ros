#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Polygon, Point32
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        self.get_logger().info("ArUco Tracking (Centre + 4 Coins + Visualisation)")

        # --- ARUCO SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # ID spécifique (None = tous)
        self.target_id = None 

        # --- ROS ---
        self.br = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  
            self.image_callback,
            10
        )

        # Publisher pour l'image debug
        self.image_pub = self.create_publisher(Image, '/aruco/result_image', 10)
        self.center_pub = self.create_publisher(Point, '/aruco/center', 10)
        self.corners_pub = self.create_publisher(Polygon, '/aruco/corners', 10)

        # --- Tracking state ---
        self.prev_center = None
        self.prev_corners = None
        self.prev_velocity = np.zeros(2)
        self.last_seen_time = None
        self.last_time = None

        # --- Parameters ---
        self.alpha = 0.7                 
        self.hold_duration = 0.5         
        self.max_jump_px = 200.0         

    def image_callback(self, msg):
        now = time.time()

        # ROS → OpenCV
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # --- Détection ArUco ---
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )

        detection_found = False
        current_center = None
        current_corners = None
        best_id_found = None

        # --- Prédiction ---
        if self.prev_center is not None and self.last_time is not None:
            dt = max(now - self.last_time, 1e-3)
            predicted_center = self.prev_center + self.prev_velocity * dt
        else:
            predicted_center = None

        # Si détection
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            candidates = []

            for i, marker_id in enumerate(ids):
                if self.target_id is not None and marker_id != self.target_id:
                    continue

                c = corners[i][0] # (4, 2)
                cx = np.mean(c[:, 0])
                cy = np.mean(c[:, 1])
                center = np.array([cx, cy])

                if predicted_center is not None:
                    dist = np.linalg.norm(center - predicted_center)
                else:
                    dist = 0.0
                
                candidates.append((i, center, dist, marker_id, c))

            # --- Sélection du meilleur candidat ---
            if len(candidates) > 0:
                candidates.sort(key=lambda x: x[2]) 
                best = candidates[0]

                if predicted_center is None or best[2] < self.max_jump_px:
                    current_center = best[1]
                    best_id_found = best[3]
                    current_corners = best[4]
                    detection_found = True

        # --- Logique Temporelle & Lissage ---
        if detection_found:
            self.last_seen_time = now

            if self.prev_center is None:
                smooth_center = current_center
                smooth_corners = current_corners
                velocity = np.zeros(2)
            else:
                dt = max(now - self.last_time, 1e-3)
                raw_velocity = (current_center - self.prev_center) / dt
                
                smooth_center = (
                    self.alpha * current_center + 
                    (1.0 - self.alpha) * self.prev_center
                )

                if self.prev_corners is not None:
                    smooth_corners = (
                        self.alpha * current_corners + 
                        (1.0 - self.alpha) * self.prev_corners
                    )
                else:
                    smooth_corners = current_corners
                
                velocity = raw_velocity

            self.prev_velocity = velocity
            self.prev_center = smooth_center
            self.prev_corners = smooth_corners

        else:
            # Mode "Hold & Predict"
            if (
                self.last_seen_time is not None and
                (now - self.last_seen_time) < self.hold_duration and
                predicted_center is not None and
                self.prev_corners is not None
            ):
                dt = max(now - self.last_time, 1e-3)
                shift = self.prev_velocity * dt
                smooth_center = self.prev_center + shift
                smooth_corners = self.prev_corners + shift

                self.prev_center = smooth_center
                self.prev_corners = smooth_corners
            else:
                self.prev_center = None
                self.prev_corners = None
                self.prev_velocity = np.zeros(2)

        self.last_time = now

        # --- Publication ROS ---
        if self.prev_center is not None:
            p_center = Point()
            p_center.x = float(self.prev_center[0])
            p_center.y = float(self.prev_center[1])
            p_center.z = float(best_id_found) if best_id_found is not None else 0.0
            self.center_pub.publish(p_center)

            if self.prev_corners is not None:
                poly_msg = Polygon()
                for i in range(4):
                    pt = Point32()
                    pt.x = float(self.prev_corners[i][0])
                    pt.y = float(self.prev_corners[i][1])
                    pt.z = 0.0
                    poly_msg.points.append(pt)
                self.corners_pub.publish(poly_msg)

        # ==========================================================
        #                 VISUALISATION (Debug Image)
        # ==========================================================
        
        # 1. Dessiner les détections brutes d'OpenCV (Bords colorés)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        # 2. Dessiner le Tracking Lissé (Centre + Coins)
        if self.prev_corners is not None and self.prev_center is not None:
            
            # --- A. Le cadre lissé (Vert) ---
            pts = self.prev_corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(cv_image, [pts], True, (0, 255, 0), 2)

            # --- B. Les 4 coins (Points Rouges + Labels) ---
            # Ordre ArUco standard : 0=TopLeft, 1=TopRight, 2=BottomRight, 3=BottomLeft
            for i, p in enumerate(self.prev_corners):
                px, py = int(p[0]), int(p[1])
                # Point rouge sur le coin
                cv2.circle(cv_image, (px, py), 5, (0, 0, 255), -1)
                # Label texte (ex: "0", "1"...)
                cv2.putText(cv_image, str(i), (px + 10, py + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            # --- C. Le Centre (Point Vert + Coordonnées) ---
            cx, cy = int(self.prev_center[0]), int(self.prev_center[1])
            
            # Cercle au centre
            cv2.circle(cv_image, (cx, cy), 6, (0, 255, 0), -1)
            
            # Texte des coordonnées (ex: "C: 320, 240")
            coord_text = f"C:{cx},{cy}"
            cv2.putText(cv_image, coord_text, (cx + 10, cy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Publication de l'image modifiée
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