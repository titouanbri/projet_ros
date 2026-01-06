import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Twist
import numpy as np
from scipy.spatial.transform import Rotation as R

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        # --- Paramètres ---
        self.declare_parameter('lambda_gain', 1.0)  # Gain de commande [cite: 49]
        self.lambda_gain = self.get_parameter('lambda_gain').value
        
        # Définition des points de l'objet (Pattern) dans son propre repère
        # Supposons un carré de 10cm x 10cm centré (comme le "pattern_points" du TP [cite: 59])
        s = 0.05 # demi-côté
        self.object_points = np.array([
            [-s, -s, 0],
            [ s, -s, 0],
            [ s,  s, 0],
            [-s,  s, 0]
        ]).T  # Shape (3, 4)

        # --- Définition de la vue désirée (s*) ---
        # On définit s* en simulant une pose désirée (ex: à 50cm face à l'objet)
        # [cite: 34, 35]
        self.s_star = self.compute_features_from_pose(
            trans_vec=np.array([0.0, 0.0, 0.5]), 
            rot_mat=np.eye(3)
        )
        self.get_logger().info(f"Features désirés (s*): {self.s_star.flatten()}")

        # --- Communication ROS ---
        # Abonnement : Pose de l'objet par rapport à la caméra (End-Effector)
        self.pose_sub = self.create_subscription(
            PoseStamped,
            '/aruco/pose', 
            self.control_loop,
            10
        )
        
        # Publication : Vitesse du End-Effector (Twist)
        self.vel_pub = self.create_publisher(Twist, '/forward_velocity_controller/commands', 10)

    def compute_features_from_pose(self, trans_vec, rot_mat):
        """
        Calcule les coordonnées normalisées (x, y) pour les 4 points.
        Correspond aux étapes de projection du TP [cite: 60-64].
        """
        features = []
        for i in range(4):
            # Point dans le repère Objet
            P_obj = np.append(self.object_points[:, i], 1) # [X, Y, Z, 1]
            
            # Transformation Monde/Objet -> Caméra A [cite: 60]
            # T_mat = [R t; 0 1]
            T_mat = np.eye(4)
            T_mat[:3, :3] = rot_mat
            T_mat[:3, 3] = trans_vec
            
            P_cam = T_mat @ P_obj # Coordonnées dans la caméra
            
            X, Y, Z = P_cam[0], P_cam[1], P_cam[2]
            
            # Coordonnées normalisées (x=X/Z, y=Y/Z) 
            # Attention à la division par zéro
            if Z <= 0.01: Z = 0.01 
            x = X / Z
            y = Y / Z
            features.extend([x, y])
            
        return np.array(features).reshape(-1, 1) # Vecteur colonne 8x1

    def compute_interaction_matrix(self, current_features, Z_estimated):
        """
        Construit la matrice L en empilant les sous-matrices Li pour chaque point.
        Formule exacte du document [cite: 16, 71-77].
        """
        L = []
        # current_features est un vecteur plat [x1, y1, x2, y2, ...]
        num_points = 4
        
        for i in range(num_points):
            x = current_features[2*i, 0]
            y = current_features[2*i+1, 0]
            Z = Z_estimated # Approximation: on utilise souvent le Z courant ou Z*
            
            # Matrice d'interaction pour un point [cite: 16]
            # [-1/Z, 0, x/Z, xy, -(1+x^2), y]
            # [0, -1/Z, y/Z, 1+y^2, -xy, -x]
            L_i = np.array([
                [-1/Z, 0,    x/Z, x*y,       -(1+x**2), y],
                [0,    -1/Z, y/Z, (1+y**2),  -x*y,      -x]
            ])
            L.append(L_i)
            
        return np.vstack(L) # Matrix 8x6

    def control_loop(self, msg):
        """
        Boucle principale [cite: 37]
        """
        # 1. Récupérer la pose actuelle (Caméra -> Objet)
        # Attention: msg.pose est la pose de l'objet vue par la caméra
        tx = msg.pose.position.x
        ty = msg.pose.position.y
        tz = msg.pose.position.z
        
        q = msg.pose.orientation
        r = R.from_quat([q.x, q.y, q.z, q.w])
        rot_mat = r.as_matrix()
        trans_vec = np.array([tx, ty, tz])

        # 2. Mesurer les points courants s(t) [cite: 38]
        s_current = self.compute_features_from_pose(trans_vec, rot_mat)

        # 3. Calculer l'erreur e = s(t) - s* [cite: 14, 39, 81]
        error = s_current - self.s_star
        
        # Critère d'arrêt simple (norme de l'erreur)
        if np.linalg.norm(error) < 0.01:
            self.publish_velocity(np.zeros(6))
            return

        # 4. Calculer la matrice d'interaction L [cite: 40, 78]
        # On utilise le Z courant de la pose pour l'estimation de profondeur
        L = self.compute_interaction_matrix(s_current, Z_estimated=tz)

        # 5. Calculer la loi de commande: v = -lambda * pseudo_inverse(L) * e
        # [cite: 15, 27, 42, 83]
        L_pinv = np.linalg.pinv(L)
        v_camera = -self.lambda_gain * (L_pinv @ error)

        # 6. Envoyer la commande au robot
        self.publish_velocity(v_camera.flatten())

    def publish_velocity(self, v_vec):
        """
        Publie le message Twist pour l'End-Effector
        v_vec = [vx, vy, vz, wx, wy, wz] dans le repère caméra
        """
        twist = Twist()
        # [cite: 102, 104]
        twist.linear.x = float(v_vec[0])
        twist.linear.y = float(v_vec[1])
        twist.linear.z = float(v_vec[2])
        twist.angular.x = float(v_vec[3])
        twist.angular.y = float(v_vec[4])
        twist.angular.z = float(v_vec[5])
        
        self.vel_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()