import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
import math  # Nécessaire pour sin()

class OscillationMover(Node):
    def __init__(self):
        super().__init__('servo_oscillator')

        self.publisher_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)
        
        self.timer_period = 0.05  # 20Hz (plus fluide pour une courbe)
        self.timer = self.create_timer(self.timer_period, self.publish_command)
        
        # On stocke le temps de départ pour calculer le sinus
        self.start_time = self.get_clock().now()

        self.get_logger().info("Démarrage de l'oscillation dans le plan X/Y...")

    def publish_command(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link' 

        # 1. Calcul du temps écoulé en secondes
        current_time = self.get_clock().now()
        elapsed_duration = current_time - self.start_time
        t = elapsed_duration.nanoseconds / 1e9  # Conversion en secondes (float)

        # 2. Paramètres du mouvement
        amplitude = 0.05  # Vitesse max en m/s (15 cm/s)
        frequence = 0.4   # Vitesse de l'oscillation (0.5 Hz = 1 cycle complet toutes les 2 sec)

        # 3. Calcul de la vitesse variable (Onde Sinusoïdale)
        # La vitesse va varier fluidement entre -0.15 et +0.15
        speed_oscillation = amplitude * math.sin(2 * math.pi * frequence * t)

        # 4. Application sur le plan (X, Y)
        # On applique la même oscillation sur X et Y -> Mouvement en diagonale
        msg.twist.linear.x = speed_oscillation
        msg.twist.linear.y = speed_oscillation
        msg.twist.linear.z = 0.0  # On reste à la même hauteur par rapport à la table

        # Pas de rotation
        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = 0.0

        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = OscillationMover()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Arrêt propre
        stop_msg = TwistStamped()
        stop_msg.header.stamp = node.get_clock().now().to_msg()
        stop_msg.header.frame_id = 'base_link'
        node.publisher_.publish(stop_msg)
        
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()