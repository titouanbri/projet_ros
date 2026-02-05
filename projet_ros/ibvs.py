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
        self.lmbda = 1.0
        self.k_orient = 2.0  # ### AJOUT ### Gain pour la correction verticale
        self.target_depth = 0.20
        self.puck_size = 0.05
        
        self.camera_frame = 'camera_color_optical_frame'
        self.tool_frame = 'tool0'
        self.base_frame = 'base_link' # ### AJOUT ### Nom de la base fixe du robot
        
        self.camera_info_topic = '/camera/camera/color/camera_info' 

        self.MAX_LIN_VEL = 0.05
        self.MAX_ANG_VEL = 0.2 # Augmenté un peu pour permettre la correction d'angle
        self.detection_timeout = 1.0 

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

        self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("--- MODE IBVS + VERTICALITE ACTIF ---")

    def camera_info_cb(self, msg):
        if self.K is None:
            self.K = np.array(msg.k).reshape(3, 3)
            self.fx = self.K[0, 0]
            self.fy = self.K[1, 1]
            self.cx = self.K[0, 2]
            self.cy = self.K[1, 2]
            
            # Forme carrée désirée
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

    def control_loop(self):
        # 1. Check Camera Info
        if self.K is None:
            return

        # 2. Check Watchdog
        now = self.get_clock().now()
        time_diff = (now - self.last_detection_time).nanoseconds / 1e9
        
        if time_diff > self.detection_timeout:
            self.stop_robot()
            return

        # 3. Check Valid Data
        if not self.data_valid:
            self.stop_robot()
            return

        # 4. Calculs
        try:
            self.compute_and_send_velocity()
        except Exception as e:
            self.get_logger().error(f"CRASH calcul : {e}")
            self.stop_robot()

    def compute_and_send_velocity(self):
        Z = self.target_depth 
        s_current = []
        L_list = []

        # --- PARTIE 1 : IBVS (Translation X, Y, Z + Rotation Z) ---
        for pt in self.current_points:
            x = (pt.x - self.cx) / self.fx
            y = (pt.y - self.cy) / self.fy
            s_current.extend([x, y])
            
            L_pt = np.array([
                [-1/Z, 0, x/Z, x*y, -(1+x**2), y],
                [ 0, -1/Z, y/Z, 1+y**2, -x*y, -x]
            ])
            L_list.append(L_pt)

        s_current = np.array(s_current)
        L = np.vstack(L_list)
        error = s_current - self.s_star

        # On garde seulement les colonnes pour Vx, Vy, Vz, Wz
        L_reduced = L[:, [0, 1, 2, 5]]
        
        # Deadband
        if np.linalg.norm(error) < 0.03:
            # Note: On continue quand même le calcul si on veut corriger l'angle même centré
            # Mais pour l'instant on s'arrête si la cible visuelle est parfaite
            self.stop_robot()
            return

        L_pinv = np.linalg.pinv(L_reduced)
        v_reduced = -self.lmbda * np.dot(L_pinv, error)

        # Construction vecteur vitesse CAMÉRA (6 DOF)
        v_cam = np.zeros(6)
        v_cam[0] = v_reduced[0] # Vx
        v_cam[1] = v_reduced[1] # Vy
        v_cam[2] = v_reduced[2] # Vz
        # v_cam[3] (Wx) et v_cam[4] (Wy) sont laissés vides pour l'instant
        v_cam[5] = v_reduced[3] # Wz (Rotation autour de l'axe optique)

        # --- PARTIE 2 : CORRECTION VERTICALE (Wx, Wy) ---
        try:
            # On cherche la TF Base -> Camera
            t_base_cam = self.tf_buffer.lookup_transform(
                self.base_frame, self.camera_frame, rclpy.time.Time())
            
            q = t_base_cam.transform.rotation
            R_bc = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

            # Axe Z de la caméra exprimé dans la base
            # (La colonne 2 de la matrice de rotation correspond à l'axe Z local)
            z_cam_in_base = R_bc[:, 2] 

            # Cible : On veut que Z caméra pointe vers le BAS du monde (0, 0, -1)
            # (Ou vers l'AVANT selon votre montage, mais standard drone/bras = bas)
            target_z = np.array([0, 0, -1])

            # Produit vectoriel pour trouver l'axe de rotation nécessaire pour aligner les vecteurs
            # axe_erreur est perpendiculaire aux deux vecteurs
            rotation_axis_in_base = np.cross(z_cam_in_base, target_z)

            # On transforme cette erreur (exprimée dans la base) vers le repère caméra
            # w_cam = R_base_cam_transpose * w_base
            rotation_error_in_cam = R_bc.T @ rotation_axis_in_base

            # Contrôleur proportionnel pour redresser
            # On injecte ça dans Wx et Wy de v_cam
            v_cam[3] = self.k_orient * rotation_error_in_cam[0]
            v_cam[4] = self.k_orient * rotation_error_in_cam[1]
            
            # Note : on touche pas à v_cam[5] qui est géré par l'IBVS

        except TransformException as ex:
            self.get_logger().warn(f"Pas de TF Base->Cam pour verticalité: {ex}", throttle_duration_sec=2)
            # On continue sans correction verticale si TF échoue

        # --- PARTIE 3 : TRANSFORMATION CAMÉRA -> OUTIL ---
        try:
            t_tool_cam = self.tf_buffer.lookup_transform(
                self.tool_frame, self.camera_frame, rclpy.time.Time())
        except TransformException as ex:
            self.stop_robot()
            return

        q_tc = t_tool_cam.transform.rotation
        t_tc = t_tool_cam.transform.translation
        R_tc = R.from_quat([q_tc.x, q_tc.y, q_tc.z, q_tc.w]).as_matrix()
        P_tc = np.array([t_tc.x, t_tc.y, t_tc.z])

        # Twist transformation: V_tool = [R  S(P)R] * V_cam
        #                       [0    R   ]
        v_c = v_cam[:3]
        w_c = v_cam[3:] # Contient maintenant Wx, Wy (verticalité) et Wz (IBVS)
        
        v_tool = (R_tc @ v_c) + np.cross(P_tc, (R_tc @ w_c))
        w_tool = R_tc @ w_c

        # Envoi Commande
        cmd = Twist()
        cmd.linear.x = self.limit_val(v_tool[0], self.MAX_LIN_VEL)
        cmd.linear.y = self.limit_val(v_tool[1], self.MAX_LIN_VEL)
        cmd.linear.z = self.limit_val(v_tool[2], self.MAX_LIN_VEL)
        cmd.angular.x = self.limit_val(w_tool[0], self.MAX_ANG_VEL)
        cmd.angular.y = self.limit_val(w_tool[1], self.MAX_ANG_VEL)
        cmd.angular.z = self.limit_val(w_tool[2], self.MAX_ANG_VEL)
        
        self.vel_pub.publish(cmd)

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