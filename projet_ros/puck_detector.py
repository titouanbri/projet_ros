#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # <--- NOUVEAU : Pour publier (u, v)
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

class CircularityDebugNode(Node):
    def __init__(self):
        super().__init__('circularity_debug_node')
        self.br = CvBridge()
        
        # Subscribers
        self.subscription = self.create_subscription(
            Image, 
            '/camera/camera/color/image_raw', 
            self.image_callback, 
            10
        )
        
        # Publishers
        self.debug_publisher_ = self.create_publisher(Image, '/puck/debug_view', 10)
        self.pos_publisher_ = self.create_publisher(Point, '/puck/position', 10) # <--- NOUVEAU
        
        # --- REGLAGES ---
        # 1. Couleur (HSV)
        self.lower_white = np.array([0, 0, 120])   
        self.upper_white = np.array([180, 50, 255]) 

        # 2. Taille (en Pixels)
        self.min_area = 100    
        self.max_area = 50000  

        # 3. Forme (Seuil de rondeur)
        self.min_circularity = 0.8 

    def image_callback(self, msg):
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError: return

        # Étape 1 : Masque Couleur
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower_white, self.upper_white)

        # Étape 2 : Nettoyage
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1) 
        mask = cv2.dilate(mask, kernel, iterations=1)

        # Étape 3 : Analyse des formes
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Variables pour trouver le meilleur candidat (le plus gros palet valide)
        best_cnt = None
        max_valid_area = 0
        best_center = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filtre de taille
            if area < self.min_area or area > self.max_area:
                continue

            # Filtre de circularité
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))

            x, y, w, h = cv2.boundingRect(cnt)

            # --- DESSIN ET LOGIQUE ---
            if circularity > self.min_circularity:
                # C'est un palet VALIDE
                color = (0, 255, 0) # Vert
                
                # Calcul des Moments pour trouver le centre exact (u, v)
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = 0, 0

                label = f"PALET ({circularity:.2f})"
                
                # On garde en mémoire si c'est le plus gros qu'on ait vu dans cette image
                if area > max_valid_area:
                    max_valid_area = area
                    best_cnt = cnt
                    best_center = (cX, cY)

            else:
                # C'est un objet rejeté (pas assez rond)
                color = (0, 0, 255) # Rouge
                label = f"NON ({circularity:.2f})"

            # Dessin de debug (Contours et Texte)
            cv2.drawContours(cv_image, [cnt], -1, color, 2)
            cv2.putText(cv_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- PUBLICATION DE LA POSITION ---
        # On ne publie que si on a trouvé au moins un palet valide
        if best_center is not None:
            point_msg = Point()
            point_msg.x = float(best_center[0]) # u (horizontal)
            point_msg.y = float(best_center[1]) # v (vertical)
            point_msg.z = 0.0                   # z (non utilisé en 2D pixel)
            self.pos_publisher_.publish(point_msg)
    
            # Dessiner une croix sur le palet élu "Cible"
            cv2.drawMarker(cv_image, best_center, (255, 0, 0), cv2.MARKER_CROSS, 20, 3)
            cv2.putText(cv_image, f"CIBLE {best_center}", (best_center[0]+10, best_center[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Publication de l'image de debug
        self.debug_publisher_.publish(self.br.cv2_to_imgmsg(cv_image, "bgr8"))

def main(args=None):
    rclpy.init(args=args)
    node = CircularityDebugNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()