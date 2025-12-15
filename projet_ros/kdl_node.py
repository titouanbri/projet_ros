import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import String 
# zob
# Messages
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

# KDL & URDF
import PyKDL
from kdl_parser_py.urdf import treeFromUrdfModel
from urdf_parser_py.urdf import URDF

class CartesianToJointVelocity(Node):
    def __init__(self):
        super().__init__('ur3_vel_ik_node')

        self.base_link = 'base_link'
        self.end_effector_link = 'tool0' # peut etre wrist_3 selon l'URDF
        
        self.kdl_tree = None
        self.kdl_chain = None
        self.ik_vel_solver = None

        #initialisation KDL
        self.get_logger().info('Chargement du modèle URDF...')
        
        self.declare_parameter('robot_description_str', '') 
        robot_desc_content = self.get_parameter('robot_description_str').value

        # Si le paramètre est vide, on essaie de charger un URDF générique (pour le test)
        # Mais dans ton cas, il faudra passer le contenu XML de l'UR3.
        if not robot_desc_content:
            self.get_logger().warn("Paramètre 'robot_description_str' vide. Le node attend l'URDF.")
            # Pour l'exemple, assure-toi de charger l'URDF correctement au lancement
            return

        robot_urdf = URDF.from_xml_string(robot_desc_content)
        ok, self.kdl_tree = treeFromUrdfModel(robot_urdf)
        
        if not ok:
            self.get_logger().error("Échec de la conversion URDF -> KDL Tree")
            return

        self.kdl_chain = self.kdl_tree.getChain(self.base_link, self.end_effector_link)
        self.num_joints = self.kdl_chain.getNrOfJoints()
        self.get_logger().info(f'Chaîne KDL créée avec {self.num_joints} joints.')

        # --- 3. Solveur KDL ---
        # Solveur de cinématique inverse pour la vitesse (Pseudo-Inverse)
        self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)

        # --- 4. Variables d'état ---
        self.current_q = PyKDL.JntArray(self.num_joints)
        self.joint_names = [] # Seront remplis via le topic joint_states

        # --- 5. Subscribers & Publishers ---
        

        self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_callback,
            qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        )
        # Abonnement à l'état actuel du robot (nécessaire pour la Jacobienne)
        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # Abonnement à la commande de vitesse cartésienne (Input)
        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel_input', # Topic d'entrée
            self.cmd_vel_callback,
            10
        )

        # Publication des vitesses articulaires calculées (Output)
        self.pub_joint_vel = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands', # Topic standard ros2_control
            10
        )


    def robot_description_callback(self, msg):
        """Ce callback n'est appelé qu'une fois, quand on reçoit l'URDF"""
        if self.kdl_tree is not None:
            return # Déjà chargé

        self.get_logger().info("URDF reçu ! Initialisation de KDL...")
        robot_urdf = URDF.from_xml_string(msg.data)
        ok, self.kdl_tree = treeFromUrdfModel(robot_urdf)
        
        if ok:
            self.kdl_chain = self.kdl_tree.getChain(self.base_link, self.end_effector_link)
            self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
            self.num_joints = self.kdl_chain.getNrOfJoints()
            # Initialiser les variables de taille dynamique maintenant
            self.current_q = PyKDL.JntArray(self.num_joints)
            self.get_logger().info("KDL Initialisé avec succès via Topic !")
        else:
            self.get_logger().error("Echec parsing URDF")

    def joint_state_callback(self, msg):
        """Met à jour la position actuelle des joints (q)"""
        # Note: Il faut mapper les noms des joints de msg vers l'ordre de KDL
        # Pour simplifier ici, on suppose que l'ordre correspond ou qu'on filtre
        # Dans un code de prod, il faut matcher msg.name avec les segments de KDL.
        
        if len(msg.position) < self.num_joints:
            return

        # Mise à jour de self.current_q
        # Attention: L'ordre des joints est crucial pour l'UR3 
        # (shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3)
        for i in range(self.num_joints):
            self.current_q[i] = msg.position[i]

    def cmd_vel_callback(self, msg):
        """Reçoit Twist, Calcule IK, Publie Joint Vels"""
        
        # 1. Convertir ROS Twist -> KDL Twist
        v_linear = PyKDL.Vector(msg.linear.x, msg.linear.y, msg.linear.z)
        v_angular = PyKDL.Vector(msg.angular.x, msg.angular.y, msg.angular.z)
        kdl_twist = PyKDL.Twist(v_linear, v_angular)

        # 2. Préparer la sortie
        q_dot_out = PyKDL.JntArray(self.num_joints)

        # 3. Résoudre : q_dot = J_pinv * twist
        ret = self.ik_vel_solver.CartToJnt(self.current_q, kdl_twist, q_dot_out)

        if ret >= 0:
            # 4. Publier le résultat
            out_msg = Float64MultiArray()
            # Convertir JntArray en liste Python
            out_msg.data = [q_dot_out[i] for i in range(self.num_joints)]
            self.pub_joint_vel.publish(out_msg)
        else:
            self.get_logger().warn("Erreur lors du calcul IK Vitesse")

def main(args=None):
    rclpy.init(args=args)
    
    # NOTE: Pour que cela fonctionne, il faut passer la description du robot
    # Habituellement, on lit le topic /robot_description ou un paramètre.
    # Ici on instancie simplement le node.
    node = CartesianToJointVelocity()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()