import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
from scipy.linalg import pinv

class IBVSController(Node):
    def __init__(self):
        super().__init__('ibvs_controller')
        
        # --- Parameters & State ---
        self.lambda_gain = 0.5
        self.s_star = None  # Goal starts as None
        self.calibrated = False
        
        # --- Variables ---
        self.bridge = CvBridge()
        self.intrinsics = None 
        self.latest_depth_img = None
        
        # --- ROS Setup ---
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        
        self.get_logger().info("IBVS Node Started. Focus the image window and press 'p' to save the goal.")

    def info_cb(self, msg):
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5]) # fx, fy, cx, cy

    def depth_cb(self, msg):
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_interaction_matrix(self, u, v, Z):
        fx, fy, cx, cy = self.intrinsics
        x, y = (u - cx) / fx, (v - cy) / fy
        L = np.array([
            [-1/Z,    0, x/Z,      x*y, -(1 + x**2),   y],
            [   0, -1/Z, y/Z, (1 + y**2),      -x*y,  -x]
        ])
        return L

    def image_cb(self, msg):
        if self.intrinsics is None or self.latest_depth_img is None:
            return

        cv_img = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Aruco Detection
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if ids is not None:
            curr_corners = corners[0].reshape(4, 2)
            s = curr_corners.flatten()
            
            # --- Visual Feedback ---
            cv2.aruco.drawDetectedMarkers(cv_img, corners, ids)
            status_text = "CALIBRATED - Tracking" if self.calibrated else "NOT CALIBRATED - Press 'p' to save goal"
            cv2.putText(cv_img, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # --- KEYBOARD INTERACTION ---
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.s_star = s.copy()
                self.calibrated = True
                self.get_logger().info(f"Goal Saved! Target Pixels: {self.s_star}")

            # --- CONTROL LOGIC ---
            if self.calibrated:
                L_stacked = []
                for i in range(4):
                    u, v = curr_corners[i]
                    # Sample depth (Z) safely
                    Z_val = self.latest_depth_img[int(v), int(u)] / 1000.0 if self.latest_depth_img[int(v), int(u)] > 0 else 0.5
                    L_stacked.append(self.get_interaction_matrix(u, v, Z_val))
                
                L = np.vstack(L_stacked)
                error = s - self.s_star
                v_camera = -self.lambda_gain * pinv(L) @ error
                self.publish_twist(v_camera)
        
        else:
            self.publish_twist(np.zeros(6))
            cv2.putText(cv_img, "Marker Lost!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("IBVS Calibration View", cv_img)
        cv2.waitKey(1)

    def publish_twist(self, v):
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = v[0], v[1], v[2]
        msg.angular.x, msg.angular.y, msg.angular.z = v[3], v[4], v[5]
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = IBVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()