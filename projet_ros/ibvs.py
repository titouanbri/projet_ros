#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Polygon
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

class IBVSNode(Node):
    def __init__(self):
        super().__init__('ibvs_node')

        # --- CONFIGURATION ---
        self.lmbda = 1
        self.target_depth = 0.20
        self.puck_size = 0.05
        
        # VERIFIEZ CES NOMS DE FRAMES
        self.camera_frame = 'camera_color_optical_frame'
        self.tool_frame = 'tool0'
        
        # VERIFIEZ CE NOM DE TOPIC (Le double /camera/camera était dans votre v1)
        self.camera_info_topic = '/camera/camera/color/camera_info' 
        # Si ça ne marche pas, essayez : '/camera/color/camera_info'

        self.MAX_LIN_VEL = 0.05
        self.MAX_ANG_VEL = 0.1
        self.detection_timeout = 1.0 # J'ai augmenté à 1s pour être plus tolérant

        # Variables internes
        self.K = None
        self.s_star = None 
        self.current_points = []
        self.last_detection_time = self.get_clock().now()
        self.data_valid = False

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subs/Pubs
        self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_cb, 1)
        self.create_subscription(Polygon, '/detected_corners', self.corners_cb, 1)
        self.vel_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 10)

        # Timer
        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("--- DEBUG MODE ACTIF ---")
        self.get_logger().info(f"Ecoute CameraInfo sur : {self.camera_info_topic}")

    def camera_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            
            w_norm = (self.puck_size / 2.0) / self.target_depth
            h_norm = (self.puck_size / 2.0) / self.target_depth
            
            self.s_star = np.array([
                -w_norm, -h_norm, w_norm, -h_norm,
                 w_norm,  h_norm, -w_norm, h_norm
            ])
            self.get_logger().info(">>> SUCCES : CameraInfo reçu et calibré !")

    def corners_cb(self, msg):
        self.current_points = msg.points
        self.last_detection_time = self.get_clock().now()
        if len(msg.points) == 4:
            self.data_valid = True
        else:
            self.data_valid = False
            self.get_logger().warn(f"Reçu {len(msg.points)} points au lieu de 4.")

    def control_loop(self):
        # 1. Check Camera Info
        if self.K is None:
            self.get_logger().info("BLOQUÉ : En attente de CameraInfo...", throttle_duration_sec=2)
            return

        # 2. Check Watchdog
        now = self.get_clock().now()
        time_diff = (now - self.last_detection_time).nanoseconds / 1e9
        
        if time_diff > self.detection_timeout:
            self.stop_robot()
            self.get_logger().info(f"BLOQUÉ : Timeout détection ({time_diff:.2f}s > {self.detection_timeout}s)", throttle_duration_sec=2)
            return

        # 3. Check Valid Data
        if not self.data_valid:
            self.stop_robot()
            self.get_logger().info("BLOQUÉ : Données invalides (pas 4 coins)", throttle_duration_sec=2)
            return

        # 4. Calculs IBVS
        try:
            self.compute_and_send_velocity()
        except Exception as e:
            self.get_logger().error(f"CRASH calcul : {e}")
            self.stop_robot()

    def compute_and_send_velocity(self):
        # ... (Logique identique à avant) ...
        Z = self.target_depth 
        s_current = []
        L_list = []

        for pt in self.current_points:
            x = (pt.x - self.cx) / self.fx
            y = (pt.y - self.cy) / self.fy
            s_current.extend([x, y])
            # Matrice d'interaction 2x6
            L_pt = np.array([
                [-1/Z, 0, x/Z, x*y, -(1+x**2), y],
                [ 0, -1/Z, y/Z, 1+y**2, -x*y, -x]
            ])
            L_list.append(L_pt)

        s_current = np.array(s_current)
        L = np.vstack(L_list)
        error = s_current - self.s_star


        # --- 4 DOF Constraint ---
        L_reduced = L[:, [0, 1, 2, 5]]
        
        # Deadband check
        if np.linalg.norm(error) < 0.03:
            self.stop_robot()
            self.get_logger().info("CIBLE ATTEINTE (Zone morte)", throttle_duration_sec=2)
            return

        L_pinv = np.linalg.pinv(L_reduced)
        v_reduced = -self.lmbda * np.dot(L_pinv, error)

        # Reconstruction 6D
        v_cam = np.zeros(6)
        v_cam[0] = v_reduced[0]
        v_cam[1] = v_reduced[1]
        v_cam[2] = v_reduced[2]
        v_cam[5] = v_reduced[3] # Rotation Z

        # --- TF Check ---
        try:
            t_tool_cam = self.tf_buffer.lookup_transform(
                self.tool_frame, self.camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.stop_robot()
            self.get_logger().warn(f"BLOQUÉ : TF introuvable {ex}", throttle_duration_sec=2)
            return

        # Transformation finale
        q = t_tool_cam.transform.rotation
        t = t_tool_cam.transform.translation
        R_tc = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
        P_tc = np.array([t.x, t.y, t.z])

        v_c = v_cam[:3]
        w_c = v_cam[3:]
        
        v_tool = (R_tc @ v_c) + np.cross(P_tc, (R_tc @ w_c))
        w_tool = R_tc @ w_c

        cmd = Twist()
        cmd.linear.x = self.limit_val(v_tool[0], self.MAX_LIN_VEL)
        cmd.linear.y = self.limit_val(v_tool[1], self.MAX_LIN_VEL)
        cmd.linear.z = self.limit_val(v_tool[2], self.MAX_LIN_VEL)
        cmd.angular.x = self.limit_val(w_tool[0], self.MAX_ANG_VEL)
        cmd.angular.y = self.limit_val(w_tool[1], self.MAX_ANG_VEL)
        cmd.angular.z = self.limit_val(w_tool[2], self.MAX_ANG_VEL)
        
        # LOG SUCCES (Throttle pour ne pas spammer)
        self.vel_pub.publish(cmd)
        self.get_logger().info(f"MOVING: Vx={cmd.linear.x:.3f}, Vy={cmd.linear.y:.3f}", throttle_duration_sec=1)

    def limit_val(self, val, limit):
        return max(min(val, limit), -limit)

    def stop_robot(self):
        self.vel_pub.publish(Twist())

def main(args=None):
    rclpy.init(args=args)
    node = IBVSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()