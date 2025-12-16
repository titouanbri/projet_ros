#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import time
import numpy as np

# Pas besoin de YOLO
# from ultralytics import YOLO

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        self.get_logger().info("ArUco Tracking (Robust Velocity + EMA)")

        # --- ARUCO SETUP ---
        # Choix du dictionnaire (4x4, 5x5, etc.)
        # Assurez-vous d'imprimer des marqueurs correspondant à ce dictionnaire
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Optionnel : Si vous voulez tracker UN SEUL ID spécifique (ex: ID 0)
        # Mettez None pour tracker n'importe quel marqueur visible
        self.target_id = None 

        # --- ROS ---
        self.br = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',  
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/aruco/result_image',
            10
        )

        self.center_pub = self.create_publisher(
            Point,
            '/aruco/center',
            10
        )

        # --- Tracking state (Identique à votre logique précédente) ---
        self.prev_center = None
        self.prev_velocity = np.zeros(2)
        self.last_seen_time = None
        self.last_time = None

        # --- Parameters ---
        self.alpha = 0.7                 # Lissage (plus haut = plus réactif, moins lisse)
        self.hold_duration = 0.5         # Temps de maintien en cas de perte de vue
        self.max_jump_px = 200.0         # Distance max pour associer la frame N à N-1

    def image_callback(self, msg):
        now = time.time()

        # ROS → OpenCV
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        # ArUco nécessite une image en niveaux de gris
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # --- Détection ArUco ---
        corners, ids, rejected = cv2.aruco.detectMarkers(
            gray, 
            self.aruco_dict, 
            parameters=self.aruco_params
        )

        detection_found = False
        current_center = None
        best_id_found = None

        # --- Prédiction (Modèle à vitesse constante) ---
        if self.prev_center is not None and self.last_time is not None:
            dt = max(now - self.last_time, 1e-3)
            predicted_center = self.prev_center + self.prev_velocity * dt
        else:
            predicted_center = None

        # Si on a détecté des marqueurs
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()
            candidates = []

            for i, marker_id in enumerate(ids):
                # Si on vise un ID spécifique, on ignore les autres
                if self.target_id is not None and marker_id != self.target_id:
                    continue

                # Calcul du centre du marqueur (moyenne des 4 coins)
                # corners[i] est de forme (1, 4, 2)
                c = corners[i][0] 
                cx = np.mean(c[:, 0])
                cy = np.mean(c[:, 1])
                center = np.array([cx, cy])

                # Calcul de la distance par rapport à la prédiction
                if predicted_center is not None:
                    dist = np.linalg.norm(center - predicted_center)
                else:
                    dist = 0.0
                
                # On stocke : (index, center, distance, id)
                candidates.append((i, center, dist, marker_id))

            # --- Sélection du meilleur candidat ---
            if len(candidates) > 0:
                # On trie par distance croissante (le plus proche de la prédiction gagne)
                candidates.sort(key=lambda x: x[2]) 
                best = candidates[0]

                # Validation : Si on a une prédiction, on vérifie qu'on n'a pas "téléporté" trop loin
                # Sauf si c'est la première détection (predicted_center is None)
                if predicted_center is None or best[2] < self.max_jump_px:
                    current_center = best[1]
                    best_id_found = best[3]
                    detection_found = True

        # --- Logique Temporelle (Identique à votre code YOLO) ---
        if detection_found:
            self.last_seen_time = now

            if self.prev_center is None:
                smooth_center = current_center
                velocity = np.zeros(2)
            else:
                dt = max(now - self.last_time, 1e-3)
                raw_velocity = (current_center - self.prev_center) / dt
                
                # Lissage de la position (EMA)
                smooth_center = (
                    self.alpha * current_center
                    + (1.0 - self.alpha) * self.prev_center
                )
                
                # On peut aussi lisser la vélocité si besoin, ici brut
                velocity = raw_velocity

            self.prev_velocity = velocity
            self.prev_center = smooth_center

        else:
            # Mode "Hold & Predict" (Occlusion partielle)
            if (
                self.last_seen_time is not None and
                (now - self.last_seen_time) < self.hold_duration and
                predicted_center is not None
            ):
                # On continue de faire confiance au modèle de mouvement
                smooth_center = predicted_center
                self.prev_center = smooth_center
                # La vélocité reste constante (inertie)
            else:
                # Perdu depuis trop longtemps
                self.prev_center = None
                self.prev_velocity = np.zeros(2)

        self.last_time = now

        # --- Publication du centre ---
        if self.prev_center is not None:
            point = Point()
            point.x = float(self.prev_center[0])
            point.y = float(self.prev_center[1])
            point.z = 0.0 # On utilise Z pour dire "ID détecté" ou 0.0
            if best_id_found is not None:
                 point.z = float(best_id_found)
            
            self.center_pub.publish(point)

        # --- Visualisation ---
        # Dessine les marqueurs bruts détectés
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        # Dessine le point tracké (lissé)
        if self.prev_center is not None:
            cv2.circle(
                cv_image,
                (int(self.prev_center[0]), int(self.prev_center[1])),
                8,
                (0, 255, 0), # Vert pour le point lissé
                -1
            )
            # Affiche l'ID tracké
            cv2.putText(
                cv_image,
                f"Tracked: {point.z:.0f}", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2
            )

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