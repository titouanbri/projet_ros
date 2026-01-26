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
        self.s_star = None  # Goal pixels
        self.calibrated = False
        
        # --- Variables ---
        self.bridge = CvBridge()
        self.intrinsics = None # (fx, fy, cx, cy)
        self.latest_depth_img = None
        
        # Initialize ArUco Detector (New API)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # --- ROS Setup ---
        # Ensure these topic names match your RealSense launch output
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.info_cb, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self.depth_cb, 10)
        
        self.get_logger().info("IBVS Node Started. Focus the image window and press 'p' to save the goal.")

    def info_cb(self, msg):
        # Extract fx, fy, cx, cy from CameraInfo K matrix
        self.intrinsics = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])

    def depth_cb(self, msg):
        # Encoding 'passthrough' for RealSense 16UC1 (millimeters)
        self.latest_depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def get_interaction_matrix(self, u, v, Z):
        if self.intrinsics is None: return np.zeros((2, 6))
        fx, fy, cx, cy = self.intrinsics
        
        # Normalized image coordinates
        x = (u - cx) / fx
        y = (v - cy) / fy
        
        # Interaction matrix for a single 2D point feature
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
        
        # Modern ArUco Detection
        corners, ids, _ = self.detector.detectMarkers(gray)

        if ids is not None:
            # Using the first marker detected
            curr_corners = corners[0].reshape(4, 2)
            s = curr_corners.flatten()
            
            # Visual Feedback
            cv2.aruco.drawDetectedMarkers(cv_img, corners, ids)
            
            # Calibration Logic (Press 'p')
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                self.s_star = s.copy()
                self.calibrated = True
                self.get_logger().info(f"Target Saved: {self.s_star}")

            if self.calibrated:
                L_stacked = []
                for i in range(4):
                    u, v = curr_corners[i]
                    
                    # Get depth Z at corner pixel (convert mm to meters)
                    # Use a small window average or center point
                    try:
                        z_raw = self.latest_depth_img[int(v), int(u)]
                        Z = z_raw / 1000.0 if z_raw > 0 else 0.5 # default 0.5m if depth lost
                    except IndexError:
                        Z = 0.5
                    
                    L_stacked.append(self.get_interaction_matrix(u, v, Z))
                
                L = np.vstack(L_stacked) # 8x6 matrix
                error = s - self.s_star
                
                # Control Law: Velocity = -Gain * PseudoInverse(L) * Error
                v_camera = -self.lambda_gain * pinv(L) @ error
                self.publish_twist(v_camera)
                
                cv2.putText(cv_img, "IBVS ACTIVE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(cv_img, "Press 'p' to set Goal", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        else:
            # Stop if marker is lost
            self.publish_twist(np.zeros(6))
            cv2.putText(cv_img, "Marker Lost", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("IBVS Monitor", cv_img)
        cv2.waitKey(1)

    def publish_twist(self, v):
        msg = Twist()
        # v = [vx, vy, vz, wx, wy, wz]
        msg.linear.x = float(v[0])
        msg.linear.y = float(v[1])
        msg.linear.z = float(v[2])
        msg.angular.x = float(v[3])
        msg.angular.y = float(v[4])
        msg.angular.z = float(v[5])
        self.cmd_pub.publish(msg)

def main():
    rclpy.init()
    node = IBVSController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()