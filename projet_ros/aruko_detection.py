#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Polygon, Point32, PoseStamped, TransformStamped
from cv_bridge import CvBridge
import cv2
import numpy as np
import math

# --- IMPORTS POUR TF ---
from tf2_ros import TransformBroadcaster

class ArucoTrackingNode(Node):
    def __init__(self):
        super().__init__('aruco_tracking_node')

        self.get_logger().info("ArUco Tracking 3D + Corners + TF (Corrigé)")

        # --- ARUCO SETUP ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # --- PARAMÈTRES 3D ---
        self.marker_length = 0.035  # Taille en mètres
        self.target_id = None 

        self.camera_matrix = None
        self.dist_coeffs = None
        

        self.br = CvBridge()
        
        # Initialisation du Broadcaster TF
        self.tf_broadcaster = TransformBroadcaster(self)
        
        #subscriptions
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
        self.corners_pub = self.create_publisher(Polygon, '/aruco/corners_pixels', 10)

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.dist_coeffs = np.array(msg.d)
            self.get_logger().info("Calibration caméra reçue !")

    def rvec_to_quaternion(self, rvec):
        """
        Convertit un vecteur de rotation (Rodrigues) d'OpenCV 
        en un Quaternion ROS [x, y, z, w]
        """
        R, _ = cv2.Rodrigues(rvec)
        
        tr = np.trace(R)
        q = [0, 0, 0, 0] # x, y, z, w

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
            
        return q # [x, y, z, w]

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
                
                # --- Récupération Translation ---
                x_trans = tvecs[i][0][0]
                y_trans = tvecs[i][0][1]
                z_trans = tvecs[i][0][2]

                # --- 1. Publication PoseStamped (Topic) ---
                pose_msg = PoseStamped()
                pose_msg.header = msg.header
                pose_msg.pose.position.x = x_trans
                pose_msg.pose.position.y = y_trans
                pose_msg.pose.position.z = z_trans
                
                # Conversion rotation
                q = self.rvec_to_quaternion(rvecs[i])
                pose_msg.pose.orientation.x = q[0]
                pose_msg.pose.orientation.y = q[1]
                pose_msg.pose.orientation.z = q[2]
                pose_msg.pose.orientation.w = q[3]

                self.pose_pub.publish(pose_msg)

                # --- 2. Publication TF (Transform) ---
                t = TransformStamped()
                
                # Header
                t.header.stamp = msg.header.stamp
                t.header.frame_id = msg.header.frame_id
                t.child_frame_id = f'aruco_{marker_id}'

                # Translation
                t.transform.translation.x = x_trans
                t.transform.translation.y = y_trans
                t.transform.translation.z = z_trans

                # Rotation (Quaternion) - CORRIGÉ ICI (rotation au lieu de orientation)
                t.transform.rotation.x = q[0]
                t.transform.rotation.y = q[1]
                t.transform.rotation.z = q[2]
                t.transform.rotation.w = q[3]

                # Envoi
                self.tf_broadcaster.sendTransform(t)
                
                # --- 3. Traitement Centre Pixel ---
                c = corners[i][0]
                cx, cy = np.mean(c[:, 0]), np.mean(c[:, 1])
                pt_msg = Point()
                pt_msg.x = float(cx)
                pt_msg.y = float(cy)
                self.center_pub.publish(pt_msg)

                # --- 4. Publication Angles ---
                poly_msg = Polygon()
                for corner in c:
                    p32 = Point32()
                    p32.x = float(corner[0])
                    p32.y = float(corner[1])
                    p32.z = 0.0
                    poly_msg.points.append(p32)
                self.corners_pub.publish(poly_msg)

                # --- VISUALISATION ---
                cv2.drawFrameAxes(cv_image, self.camera_matrix, self.dist_coeffs, rvecs[i], tvecs[i], 0.03)
                
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