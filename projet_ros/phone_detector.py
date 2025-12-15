#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point  # 1. Import du message Point
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time 
import numpy as np

from ultralytics import YOLO

class YoloSmartphoneNode(Node):
    def __init__(self):
        super().__init__('yolo_smartphone_node')

        self.get_logger().info("Initialisation YOLOv8 pour détection de smartphones...")
        
        self.model = YOLO("yolov8n-seg.pt")  
        
        # Ros config
        self.br = CvBridge()

        # Subscriber
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, 10)
        
        # Publisher Image (Visualisation)
        self.publisher_ = self.create_publisher(Image, '/yolo/smartphone_result', 10)
        
        # 2. Publisher Coordonnées (Centre)
        self.center_publisher_ = self.create_publisher(Point, '/yolo/smartphone_center', 10)

    def image_callback(self, msg):
        start_total = time.time()
        
        # Conversion ROS Image -> OpenCV
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(f"Error CvBridge: {e}")
            return

        # Inférence 
        start_inference = time.time()
        
        # Detection (classes=[67] pour 'cell phone' dans COCO)
        results = self.model.predict(source=cv_image, verbose=False, classes=[67], conf=0.5)
        
        end_inference = time.time()

        # Visualisation de base YOLO (dessine les boîtes)
        res_image = results[0].plot() 

        # 3. Traitement des résultats pour trouver le centre
        # results[0].boxes contient toutes les boîtes détectées
        boxes = results[0].boxes
        
        if boxes is not None:
            for box in boxes:
                # Récupération des coordonnées (x1, y1, x2, y2)
                # .cpu().numpy() permet de convertir le tensor PyTorch en tableau numpy
                coords = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = coords
                
                # Calcul du centre
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                # Création du message Point
                point_msg = Point()
                point_msg.x = float(center_x)
                point_msg.y = float(center_y)
                point_msg.z = 0.0 # 0 car c'est une image 2D
                
                # Publication
                self.center_publisher_.publish(point_msg)
                
                # Optionnel : Dessiner un point rouge au centre sur l'image de retour
                cv2.circle(res_image, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)
                self.get_logger().info(f"Smartphone détecté au centre: x={center_x:.1f}, y={center_y:.1f}")

        # Conversion OpenCV -> ROS Image et Publication Visualisation
        out_msg = self.br.cv2_to_imgmsg(res_image, "bgr8")
        self.publisher_.publish(out_msg)

        end_total = time.time()
        
        # Logging FPS
        total_time = end_total - start_total
        fps = 1.0 / total_time if total_time > 0 else 0.0
        # self.get_logger().info(f"FPS: {fps:.2f}") # Décommenter si besoin

def main(args=None):
    rclpy.init(args=args)

    try:
        node = YoloSmartphoneNode()
        print("Noeud YOLO Smartphone démarré. Publication des centres sur '/yolo/smartphone_center'.")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Erreur: {e}")
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()