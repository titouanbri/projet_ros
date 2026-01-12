#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped
from scipy.spatial.transform import Rotation as R
import numpy as np

class PBVSNode(Node):
    def __init__(self):
        super().__init__('pbvs_node')

        # --- PARAMÈTRES ---
        self.lmbda = 1.0        # Gain proportionnel (lambda)
        self.dist_target = 0.3  # Distance désirée (30 cm)
        
        # --- SÉCURITÉ UR3 ---
        self.MAX_LIN_VEL = 0.05  # m/s
        self.MAX_ANG_VEL = 0.5  # rad/s
        
        # Définition de la Pose Désirée du Marqueur dans la Caméra (T_des)
        # On veut le marqueur à 'dist_target' devant la caméra (axe Z).
        # Orientation : Le repère Aruco a Z sortant. La caméra a Z devant.
        # Pour faire face, il faut une rotation de 180° autour de X (ou Y) pour opposer les Z.
        # Matrice de rotation pour 180° autour de X :
        rot_target = R.from_euler('x', 180, degrees=True).as_matrix()
        pos_target = np.array([0.0, 0.0, self.dist_target])
        
        self.T_des = np.eye(4)
        self.T_des[:3, :3] = rot_target
        self.T_des[:3, 3] = pos_target

        # Publishers & Subscribers
        self.pose_sub = self.create_subscription(
            PoseStamped, 
            '/aruco/pose', 
            self.control_loop, 
            10
        )
        self.vel_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 10)
        
        self.get_logger().info("Nœud PBVS (Position Based Visual Servoing) démarré.")
        self.get_logger().info(f"Cible : Marqueur à {self.dist_target}m en face.")

    def transform_from_pose(self, pose):
        """Convertit geometry_msgs/Pose en matrice 4x4"""
        t = np.array([pose.position.x, pose.position.y, pose.position.z])
        q = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat(q).as_matrix()
        mat[:3, 3] = t
        return mat

    def limit_velocity(self, v, max_val):
        """Sature la vitesse en gardant la direction"""
        norm = np.linalg.norm(v)
        if norm > max_val:
            return v * (max_val / norm)
        return v

    def control_loop(self, msg):
        # 1. Récupération de la pose actuelle (T_curr = Marker dans Caméra)
        T_curr = self.transform_from_pose(msg.pose)        
        X = self.T_des @ np.linalg.inv(T_curr)

        # Extraction Rotation (R) et Translation (t) de l'erreur
        R_mat = X[:3, :3]
        t_vec = X[:3, 3]

        # 3. Calcul de l'axe-angle (theta * u) pour la rotation
        r = R.from_matrix(R_mat)
        rot_vec = r.as_rotvec() # Cela correspond à (theta * u)

        # 4. Loi de Commande (Formule (5) du TP page 2)
        # v = -lambda * R^T * t
        # omega = -lambda * theta * u
        
        # Note: R^T * t ramène le vecteur translation du repère B au repère A (Caméra actuelle)
        # C'est nécessaire car on doit envoyer une vitesse exprimée dans le repère courant de la caméra.
        
        v_lin = -self.lmbda * (R_mat.T @ t_vec)
        v_ang = -self.lmbda * rot_vec

        # 5. Sécurité et Saturation
        v_lin = self.limit_velocity(v_lin, self.MAX_LIN_VEL)
        v_ang = self.limit_velocity(v_ang, self.MAX_ANG_VEL)

        # 6. Envoi de la commande
        cmd = Twist()
        # On inverse les axes si nécessaire selon le repère de contrôle du robot (ivk.py)
        # Généralement ivk.py attend x=devant/droite, mais la caméra a z=devant.
        # Si la caméra est montée telle quelle sur l'effecteur, le repère caméra est :
        # Z (optique) aligné avec Z (effecteur) ? A vérifier.
        # Souvent : Z_cam = X_ee ou Z_ee. 
        # Ici on envoie la commande dans le repère "Caméra". 
        # Si ivk.py contrôle l'effecteur et que la caméra est l'effecteur, c'est bon.
        
        cmd.linear.x = float(v_lin[0])
        cmd.linear.y = float(v_lin[1])
        cmd.linear.z = float(v_lin[2])
        cmd.angular.x = float(v_ang[0])
        cmd.angular.y = float(v_ang[1])
        cmd.angular.z = float(v_ang[2])

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