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

from ultralytics import YOLO


class YoloSmartphoneNode(Node):
    def __init__(self):
        super().__init__('yolo_smartphone_node')

        self.get_logger().info("YOLO Smartphone Tracking (fast-motion robust)")

        # --- YOLO ---
        self.model = YOLO("models/yolo11s.pt")

        # --- ROS ---
        self.br = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw', #/webcam/image/raw
            self.image_callback,
            10
        )

        self.image_pub = self.create_publisher(
            Image,
            '/yolo/smartphone_result',
            10
        )

        self.center_pub = self.create_publisher(
            Point,
            '/yolo/smartphone_center',
            10
        )

        # --- Tracking state ---
        self.active_track_id = None
        self.prev_center = None
        self.prev_velocity = np.zeros(2)
        self.last_seen_time = None
        self.last_time = None

        # --- Parameters (industry typical) ---
        self.alpha = 0.65                # EMA smoothing
        self.hold_duration = 0.5         # seconds
        self.conf_threshold = 0.4        # lower for fast motion
        self.max_jump_px = 150.0         # motion gate (pixels)

    def image_callback(self, msg):
        now = time.time()

        # ROS → OpenCV
        cv_image = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # --- YOLO tracking ---
        results = self.model.track(
            source=cv_image,
            persist=True,
            classes=[67],
            conf=self.conf_threshold,
            verbose=False
        )

        res = results[0]
        boxes = res.boxes

        detection_found = False
        current_center = None

        # --- Prediction (constant velocity model) ---
        if self.prev_center is not None and self.last_time is not None:
            dt = max(now - self.last_time, 1e-3)
            predicted_center = self.prev_center + self.prev_velocity * dt
        else:
            predicted_center = None

        if boxes is not None and len(boxes) > 0 and boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            candidates = []

            for i in range(len(track_ids)):
                x1, y1, x2, y2 = xyxy[i]
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0
                center = np.array([cx, cy])

                if predicted_center is not None:
                    dist = np.linalg.norm(center - predicted_center)
                else:
                    dist = 0.0

                candidates.append((i, center, dist, confs[i], track_ids[i]))

            # --- Choose best candidate ---
            candidates.sort(key=lambda x: (x[2], -x[3]))  # distance first, then confidence
            best = candidates[0]

            if predicted_center is None or best[2] < self.max_jump_px:
                current_center = best[1]
                self.active_track_id = best[4]
                detection_found = True

        # --- Temporal logic ---
        if detection_found:
            self.last_seen_time = now

            if self.prev_center is None:
                smooth_center = current_center
                velocity = np.zeros(2)
            else:
                dt = max(now - self.last_time, 1e-3)
                velocity = (current_center - self.prev_center) / dt
                smooth_center = (
                    self.alpha * current_center
                    + (1.0 - self.alpha) * self.prev_center
                )

            self.prev_velocity = velocity
            self.prev_center = smooth_center

        else:
            # Hold & predict
            if (
                self.last_seen_time is not None and
                (now - self.last_seen_time) < self.hold_duration and
                predicted_center is not None
            ):
                smooth_center = predicted_center
                self.prev_center = smooth_center
            else:
                self.active_track_id = None
                self.prev_center = None
                self.prev_velocity = np.zeros(2)

        self.last_time = now

        # --- Publish center if available ---
        if self.prev_center is not None:
            point = Point()
            point.x = float(self.prev_center[0])
            point.y = float(self.prev_center[1])
            point.z = 0.0
            self.center_pub.publish(point)

        # --- Always publish visualization ---
        vis = res.plot()

        if self.prev_center is not None:
            cv2.circle(
                vis,
                (int(self.prev_center[0]), int(self.prev_center[1])),
                6,
                (0, 0, 255),
                -1
            )

        out_msg = self.br.cv2_to_imgmsg(vis, encoding='bgr8')
        self.image_pub.publish(out_msg)


def main(args=None):
    rclpy.init(args=args)
    node = YoloSmartphoneNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
