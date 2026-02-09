#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Bool
from cv_bridge import CvBridge
import cv2
import numpy as np


class CameraCalibNode(Node):
    def __init__(self):
        super().__init__('camera_calib_node')

        self.bridge = CvBridge()

        # --- capture trigger ---
        self.capture_requested = False
        self.create_subscription(Bool, '/camera_calib_capture', self.capture_cb, 1)

        self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_cb,
            1
        )

        self.cam_info_pub = self.create_publisher(
            CameraInfo,
            '/camera_info_calibrated',
            1
        )

        # --- ArUco board ---
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )
        self.board = cv2.aruco.GridBoard(
            size=(5, 7),
            markerLength=0.015,
            markerSeparation=0.004,
            dictionary=self.aruco_dict
        )
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict)

        # --- storage ---
        self.all_corners = []
        self.all_ids = []
        self.image_size = None
        self.required_views = 9

    def capture_cb(self, msg: Bool):
        self.capture_requested = msg.data

    def image_cb(self, msg: Image):
        if not self.capture_requested:
            return

        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception:
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is None or len(ids) < 6:
            return

        if self.image_size is None:
            self.image_size = gray.shape[::-1]

        self.all_corners.append(corners)
        self.all_ids.append(ids)
        self.capture_requested = False

        self.get_logger().info(
            f"Views collected: {len(self.all_corners)}/{self.required_views}"
        )

        if len(self.all_corners) >= self.required_views:
            self.calibrate()

    def calibrate(self):
        obj_points = []
        img_points = []

        for corners, ids in zip(self.all_corners, self.all_ids):
            objp, imgp = self.board.matchImagePoints(corners, ids)
            if objp is None or imgp is None:
                continue
            obj_points.append(objp)
            img_points.append(imgp)

        if len(obj_points) < 3:
            self.get_logger().error("Not enough valid views for calibration")
            return

        ret, K, D, _, _ = cv2.calibrateCamera(
            objectPoints=obj_points,
            imagePoints=img_points,
            imageSize=self.image_size,
            cameraMatrix=None,
            distCoeffs=None
        )

        if not ret:
            self.get_logger().error("Camera calibration failed")
            return

        cam_info = CameraInfo()
        cam_info.width = self.image_size[0]
        cam_info.height = self.image_size[1]
        cam_info.k = K.flatten().tolist()
        cam_info.d = D.flatten().tolist()
        cam_info.r = np.eye(3).flatten().tolist()
        cam_info.p = np.hstack([K, np.zeros((3, 1))]).flatten().tolist()
        cam_info.distortion_model = 'plumb_bob'

        self.cam_info_pub.publish(cam_info)
        self.get_logger().info("Camera intrinsics published")



def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
