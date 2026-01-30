#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point, Polygon, Point32
from cv_bridge import CvBridge
import cv2
import numpy as np
import os 
from ament_index_python.packages import get_package_share_directory 

from ultralytics import YOLO

class DetectionNode(Node):
    def __init__(self):
        super().__init__('detection_node_titou')

        self.get_logger().info("Detection Node initialized (Raw Mode - No Prediction)")
        
        # Chargement du modèle
        package_share_directory = get_package_share_directory('projet_ros')
        # model_path = os.path.join(package_share_directory, 'models', 'puck_detector_n.pt')
        model_path = os.path.join(package_share_directory, 'models', 'puck_detector_n_openvino_model')

        try:
            # self.model = YOLO(model_path)
            self.model = YOLO(model_path, task='detect')

        except Exception as e:
            self.get_logger().error(f"Impossible de charger le modèle : {e}")
            raise e

        # ROS 
        self.br = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            3
        )

        self.image_pub = self.create_publisher(
            Image,
            '/detection_results',
            10
        )

        self.center_pub = self.create_publisher(
            Point,
            '/detected_center',
            10
        )

        self.corners_pub = self.create_publisher(
            Polygon,
            '/detected_corners',
            10
        )

        # Configuration YOLO
        self.conf_threshold = 0.8

    def image_callback(self, msg):
        # ROS → OpenCV
        try:
            cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Erreur conversion image: {e}")
            return

        # YOLO tracking
        results = self.model.track(
            source=cv_image,
            persist=True,
            classes=[0], 
            conf=self.conf_threshold,
            verbose=False,
            device='cpu'
        )

        res = results[0]
        boxes = res.boxes
        vis = res.plot() # Dessine la boite YOLO sur l'image

        target_center = None
        target_wh = None

        # Si une détection existe
        if boxes is not None and len(boxes) > 0:
            # Récupération des données
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            # On prend simplement la détection avec la plus haute confiance
            # (Puisqu'on a enlevé la prédiction, on ne filtre plus par distance prédite)
            best_idx = np.argmax(confs)
            
            x1, y1, x2, y2 = xyxy[best_idx]
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            
            target_center = np.array([cx, cy])
            target_wh = np.array([w, h])

        # --- PUBLICATION (Uniquement si détecté actuellement) ---
        if target_center is not None:
            # 1. Publier le centre brut
            point = Point()
            point.x = float(target_center[0])
            point.y = float(target_center[1])
            point.z = 0.0
            self.center_pub.publish(point)

            # Dessin du centre sur l'image de sortie (point vert)
            cv2.circle(vis, (int(target_center[0]), int(target_center[1])), 6, (0, 255, 0), -1)

            # 2. Calculer et publier les 4 coins
            cx, cy = target_center
            w, h = target_wh
            
            corners = [
                (cx - w/2, cy - h/2), # TL
                (cx + w/2, cy - h/2), # TR
                (cx + w/2, cy + h/2), # BR
                (cx - w/2, cy + h/2)  # BL
            ]

            poly_msg = Polygon()
            for (x, y) in corners:
                p = Point32()
                p.x = float(x)
                p.y = float(y)
                p.z = 0.0
                poly_msg.points.append(p)
            
            self.corners_pub.publish(poly_msg)

        # Publication de l'image (avec ou sans détection)
        out_msg = self.br.cv2_to_imgmsg(vis, encoding='bgr8')
        self.image_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()