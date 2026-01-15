#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, TransformStamped
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
import numpy as np

class PBVSNode(Node):
    def __init__(self):
        super().__init__('pbvs_node')

        # --- PARAMÈTRES ---
        self.lmbda = 1.0        # Gain proportionnel (lambda)
        self.dist_target = 0.3  # Distance désirée (30 cm)
        
        # Noms des frames TF
        # 'aruco_0' est l'ID par défaut. Si vous utilisez un autre ID, changez ce paramètre.
        self.target_frame = 'aruco_0'
        self.camera_frame = 'camera_link' 
        self.tool_frame = 'tool0'
        
        # --- SÉCURITÉ UR3 ---
        self.MAX_LIN_VEL = 0.05  # m/s
        self.MAX_ANG_VEL = 0.5   # rad/s
        
        # Définition de la Pose Désirée du Marqueur dans la Caméra (T_des)
        # Identique à avant : Marqueur à 30cm devant, Z opposés.
        rot_target = R.from_euler('x', -180, degrees=True).as_matrix()
        pos_target = np.array([0.0, 0.0, self.dist_target])
        
        self.T_des = np.eye(4)
        self.T_des[:3, :3] = rot_target
        self.T_des[:3, 3] = pos_target

        # --- SETUP TF ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher vitesse
        self.vel_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 10)
        
        # Timer de contrôle (10 Hz)
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("node launched")

    def transform_to_matrix(self, t_stamped):
        """Convertit un message Geometry/Transform en matrice Numpy 4x4"""
        t = t_stamped.transform.translation
        r = t_stamped.transform.rotation
        
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def limit_velocity(self, v, max_val):
        """Sature la vitesse en gardant la direction"""
        norm = np.linalg.norm(v)
        if norm > max_val:
            return v * (max_val / norm)
        return v

    def control_loop(self):
        # 1. Récupération de la Pose Actuelle via TF (Camera -> Marker)
        try:
            # On cherche la transformation du repère Caméra VERS le Marqueur
            # Cela correspond à la pose du marqueur dans la caméra.
            t_cam_marker = self.tf_buffer.lookup_transform(
                self.camera_frame,      # Target frame (Reference)
                self.target_frame,      # Source frame (Object)
                rclpy.time.Time()
            )
            # Vérification de l'âge de la TF (eviter de garder de garder un ghost)
            now = self.get_clock().now()
            # Temps de la TF reçue
            tf_time = rclpy.time.Time.from_msg(t_cam_marker.header.stamp)
            # Différence en secondes
            age = (now - tf_time).nanoseconds / 1e9
            
            if age > 0.5:
                # La donnée est trop vieille (> 0.5s), le marqueur n'est probablement plus là
                # self.get_logger().warn(f"TF trop vieille : {age:.2f}s")
                self.vel_pub.publish(Twist()) # STOP
                return
        except TransformException as ex:
            # Si la TF n'est pas dispo (ex: marqueur non visible), on arrête.
            # self.get_logger().warn(f'Pas de TF marker: {ex}')
            self.vel_pub.publish(Twist()) # Stop
            return

        # Conversion en matrice T_curr
        T_curr = self.transform_to_matrix(t_cam_marker)
        #on ingore la rotation 
        T_curr[:3, :3] = self.T_des[:3, :3]
        # 2. Calcul de l'erreur dans le repère CAMÉRA
        # X = T_des * inv(T_curr)
        X = self.T_des @ np.linalg.inv(T_curr)

        R_mat = X[:3, :3]
        t_vec = X[:3, 3]

        r_obj = R.from_matrix(R_mat)
        rot_vec = r_obj.as_rotvec()

        # Loi de commande PBVS (v_cam, w_cam sont exprimés dans le repère Caméra)
        v_cam = -self.lmbda * (R_mat.T @ t_vec)*(-1)
        w_cam = -self.lmbda * rot_vec*(-1)

        # 3. Correction cinématique : Passage du repère Caméra au repère Effecteur (Tool)
        # On a besoin de la TF Tool -> Camera pour savoir comment la caméra est montée
        try:
            t_tool_cam = self.tf_buffer.lookup_transform(
                self.tool_frame,
                self.camera_frame,
                rclpy.time.Time()
            )
        except TransformException as ex:
            self.get_logger().error(f'TF Tool->Cam manquante : {ex}')
            self.vel_pub.publish(Twist())
            return

        # Matrice de transformation Tool -> Cam
        T_tc = self.transform_to_matrix(t_tool_cam)
        R_tc = T_tc[:3, :3] # Rotation
        P_tc = T_tc[:3, 3]  # Translation (Bras de levier)

        # Transformation de la vitesse (Adjoint Map / Rigid Body Velocity)
        # On veut V_tool tel que la caméra bouge à V_cam.
        # Formule : V_tool = V_cam_in_tool + P_tc x W_cam_in_tool
        
        # D'abord, on tourne les vecteurs vitesse pour les aligner avec Tool
        v_cam_in_tool = R_tc @ v_cam
        w_cam_in_tool = R_tc @ w_cam
        
        # Ensuite on applique le produit vectoriel pour le bras de levier
        # v_tool = v_cam (rot) + cross(P_tc, w_cam (rot))
        v_tool = v_cam_in_tool + np.cross(P_tc, w_cam_in_tool)
        w_tool = w_cam_in_tool

        # 4. Saturation et Envoi
        v_tool = self.limit_velocity(v_tool, self.MAX_LIN_VEL)
        w_tool = self.limit_velocity(w_tool, self.MAX_ANG_VEL)

        cmd = Twist()
        cmd.linear.x = float(v_tool[0])
        cmd.linear.y = float(v_tool[1])
        cmd.linear.z = float(v_tool[2])
        cmd.angular.x = float(w_tool[0])
        cmd.angular.y = float(w_tool[1])
        cmd.angular.z = float(w_tool[2])

        self.vel_pub.publish(cmd)

def main(args=None):
    rclpy.init(args=args)
    node = PBVSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Arrêt propre
        stop = Twist()
        node.vel_pub.publish(stop)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()