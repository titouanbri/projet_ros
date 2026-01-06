import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Polygon 
import numpy as np
import time

class VisualServoingNode(Node):
    def __init__(self):
        super().__init__('visual_servoing_node')

        # --- 1. Paramètres de la Caméra ---
        self.fx = 800.0
        self.fy = 800.0
        self.u0 = 320.0
        self.v0 = 240.0
        
        # --- 2. Configuration de l'algorithme ---
        self.lmbda = 0.5 
        self.Z_est = 1.0 

        # --- SECURITE UR3 (Nouveaux paramètres) ---
        # Limites strictes pour un UR3 (valeurs conservatrices pour commencer)
        self.MAX_LIN_VEL = 0.03    # m/s (10 cm/s)
        self.MAX_ANG_VEL = 0.06    # rad/s
        self.MAX_LIN_ACC = 0.05   # m/s^2 (Accélération max par pas de temps)
        self.MAX_ANG_ACC = 0.1    # rad/s^2
        
        # Mémoire pour le lissage de l'accélération
        self.prev_linear = np.zeros(3)
        self.prev_angular = np.zeros(3)
        self.last_time = self.get_clock().now()

        # --- 3. Définition de la Cible (s*) ---
        target_pixels = [
            (220, 140), 
            (420, 140), 
            (420, 340), 
            (220, 340) 
        ]
        
        self.s_star = []
        for (u, v) in target_pixels:
            x, y = self.pixel_to_normalized(u, v)
            self.s_star.extend([x, y])
        self.s_star = np.array(self.s_star).reshape(-1, 1)

        # --- 4. ROS Publishers & Subscribers ---
        self.publisher_vel = self.create_publisher(Twist, '/ee_velocity_cmd', 10)
        
        self.subscription = self.create_subscription(
            Polygon,
            '/aruco/corners_pixels', 
            self.control_loop,
            10) # 10Hz -> dt approx 0.1s
            
        self.get_logger().info("Visual Servoing Node Initialized with Safety Limits.")

    def pixel_to_normalized(self, u, v):
        x = (u - self.u0) / self.fx
        y = (v - self.v0) / self.fy
        return x, y

    def compute_interaction_matrix_point(self, x, y, Z):
        L_i = np.array([
            [-1.0/Z,  0.0,    x/Z,      x*y,       -(1 + x**2),  y],
            [ 0.0,   -1.0/Z,  y/Z,      1 + y**2,  -x*y,         -x]
        ])
        return L_i

    def limit_velocity_vector(self, v_vector, max_val):
        """
        Reduit la norme du vecteur sans changer sa direction
        """
        norm = np.linalg.norm(v_vector)
        if norm > max_val:
            scale = max_val / norm
            return v_vector * scale
        return v_vector

    def limit_acceleration(self, target_v, prev_v, max_acc, dt):
        """
        Limite le changement de vitesse (accélération)
        """
        diff = target_v - prev_v
        max_change = max_acc * dt # Delta V max autorisé pour ce pas de temps
        
        # Si le changement est trop brusque, on le cape
        diff_norm = np.linalg.norm(diff)
        if diff_norm > max_change:
            diff = diff * (max_change / diff_norm)
            
        return prev_v + diff

    def control_loop(self, msg):
        # Gestion du temps pour l'accélération
        current_time = self.get_clock().now()
        dt = (current_time - self.last_time).nanoseconds / 1e9
        if dt == 0: dt = 0.1 # Sécurité division par zéro
        self.last_time = current_time

        if len(msg.points) != 4:
            self.get_logger().warn(f"Mauvais nombre de points: {len(msg.points)}")
            # En cas de perte de tracking, on envoie STOP par sécurité
            stop_msg = Twist()
            self.publisher_vel.publish(stop_msg)
            return

        # 1. & 2. Mesure et Interaction
        current_features = []
        L_stack = [] 
        
        for point in msg.points:
            x, y = self.pixel_to_normalized(point.x, point.y)
            current_features.extend([x, y])
            L_i = self.compute_interaction_matrix_point(x, y, self.Z_est)
            L_stack.append(L_i)

        s_current = np.array(current_features).reshape(-1, 1)
        L = np.vstack(L_stack)

        # 3. Calcul de l'erreur
        error = s_current - self.s_star

        # 4. Loi de commande brute
        try:
            L_pinv = np.linalg.pinv(L)
        except np.linalg.LinAlgError:
            self.get_logger().error("Erreur inversion matrice")
            return

        velocity_raw = -self.lmbda * np.dot(L_pinv, error)
        
        # Séparation Linéaire / Angulaire
        v_lin_raw = velocity_raw[0:3].flatten()
        v_ang_raw = velocity_raw[3:6].flatten()

        # --- 5. APPLICATION DES LIMITES DE SECURITE ---

        # A. Saturation de Vitesse (Scaling)
        # On s'assure que le vecteur ne dépasse pas MAX_LIN_VEL tout en gardant la direction
        v_lin_clamped = self.limit_velocity_vector(v_lin_raw, self.MAX_LIN_VEL)
        v_ang_clamped = self.limit_velocity_vector(v_ang_raw, self.MAX_ANG_VEL)

        # B. Limitation d'Accélération (Smoothing)
        # On lisse la transition entre la vitesse précédente et la nouvelle cible
        v_lin_final = self.limit_acceleration(v_lin_clamped, self.prev_linear, self.MAX_LIN_ACC, dt)
        v_ang_final = self.limit_acceleration(v_ang_clamped, self.prev_angular, self.MAX_ANG_ACC, dt)

        # Mise à jour de la mémoire pour la prochaine boucle
        self.prev_linear = v_lin_final
        self.prev_angular = v_ang_final

        # 6. Publication
        cmd_msg = Twist()
        cmd_msg.linear.x = v_lin_final[0]
        cmd_msg.linear.y = v_lin_final[1]
        cmd_msg.linear.z = v_lin_final[2]
        cmd_msg.angular.x = v_ang_final[0]
        cmd_msg.angular.y = v_ang_final[1]
        cmd_msg.angular.z = v_ang_final[2]

        self.publisher_vel.publish(cmd_msg)
        
        # Debug optionnel
        # error_norm = np.linalg.norm(error)
        # self.get_logger().info(f"Err: {error_norm:.3f} | V_lin: {np.linalg.norm(v_lin_final):.3f}")

def main(args=None):
    rclpy.init(args=args)
    node = VisualServoingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # Arrêt propre en cas de CTRL+C
        stop_msg = Twist()
        node.publisher_vel.publish(stop_msg)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()