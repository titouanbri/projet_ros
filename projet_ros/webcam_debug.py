#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2

from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import Header
from cv_bridge import CvBridge


class WebcamDualPublisher(Node):
    def __init__(self):
        super().__init__('webcam_dual_publisher')
        self.debug = True

        # RAW image publisher
        self.raw_pub = self.create_publisher(
            Image,
            '/webcam/image/raw',
            10
        )

        # JPEG compressed publisher
        self.compressed_pub = self.create_publisher(
            CompressedImage,
            '/webcam/image/compressed',
            10
        )

        self.debug_pub = self.create_publisher(
            Image,
            '/detection_results',
            10
        )

        self.bridge = CvBridge()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Could not open webcam')
            raise RuntimeError('Webcam not accessible')

        self.timer = self.create_timer(1.0 / 60.0, self.timer_callback)
        self.get_logger().info('Publishing RAW and JPEG webcam images')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning('Failed to grab frame')
            return

        stamp = self.get_clock().now().to_msg()

        # -------- RAW IMAGE --------
        raw_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        raw_msg.header.stamp = stamp
        raw_msg.header.frame_id = 'webcam'
        self.raw_pub.publish(raw_msg)
        if self.debug:
            self.debug_pub.publish(raw_msg)

        # -------- JPEG COMPRESSED --------
        success, encoded = cv2.imencode(
            '.jpg',
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        )
        if not success:
            self.get_logger().warning('JPEG encoding failed')
            return

        comp_msg = CompressedImage()
        comp_msg.header = Header()
        comp_msg.header.stamp = stamp
        comp_msg.header.frame_id = 'webcam'
        comp_msg.format = 'jpeg'
        comp_msg.data = encoded.tobytes()

        self.compressed_pub.publish(comp_msg)


def main():
    rclpy.init()
    node = WebcamDualPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
