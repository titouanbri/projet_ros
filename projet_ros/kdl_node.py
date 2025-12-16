import sys
import rclpy
import time
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState

# KDL & URDF
import PyKDL
from urdf_parser_py.urdf import URDF

def urdf_to_kdl(robot_model):
    """Convertit un modèle URDF en arbre KDL"""
    tree = PyKDL.Tree(robot_model.get_root())
    
    def add_children_to_tree(curr_link_name):
        if curr_link_name in robot_model.child_map:
            for child_joint_name, child_link_name in robot_model.child_map[curr_link_name]:
                joint = robot_model.joint_map[child_joint_name]
                j_type = str(joint.type).strip().lower()

                if j_type == 'revolute' or j_type == 'continuous':
                    kdl_joint = PyKDL.Joint(joint.name, PyKDL.Joint.RotZ)
                elif j_type == 'prismatic':
                    kdl_joint = PyKDL.Joint(joint.name, PyKDL.Joint.TransZ)
                else:
                    kdl_joint = PyKDL.Joint(joint.name)

                origin = joint.origin
                kdl_origin = PyKDL.Frame(
                    PyKDL.Rotation.RPY(origin.rpy[0], origin.rpy[1], origin.rpy[2]),
                    PyKDL.Vector(origin.xyz[0], origin.xyz[1], origin.xyz[2])
                )

                kdl_segment = PyKDL.Segment(child_link_name, kdl_joint, kdl_origin)
                tree.addSegment(kdl_segment, curr_link_name)
                add_children_to_tree(child_link_name)

    add_children_to_tree(robot_model.get_root())
    return True, tree

class CartesianToJointVelocity(Node):
    def __init__(self):
        super().__init__('ur3_vel_ik_node')

        # Configuration
        self.possible_base_links = ['base_link_inertia', 'base_link', 'base', 'world']
        self.end_effector_link = 'tool0' 
        
        self.kdl_tree = None
        self.kdl_chain = None
        self.ik_vel_solver = None
        self.num_joints = 0
        self.joint_names = [] # Pour stocker l'ordre exact de la chaîne KDL
        self.urdf_xml = None
        self.kdl_initialized = False

        # Variables de contrôle
        self.target_twist = PyKDL.Twist() # Vitesse désirée (0 par défaut)
        self.last_cmd_time = 0.0          # Pour le watchdog
        self.cmd_timeout = 0.5            # Arrêt après 0.5s sans commande
        self.current_joint_positions = {} # Dictionnaire {nom: position}

        # --- Initialisation des Paramètres ---
        self.declare_parameter('robot_description_str', '') 
        param_desc = self.get_parameter('robot_description_str').value
        if param_desc:
            self.urdf_xml = param_desc

        # --- Subscribers & Publishers ---
        self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_callback,
            qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        )
        
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.create_subscription(
            Twist,
            '/cmd_vel_input', # Topic d'entrée
            self.cmd_vel_callback,
            10
        )

        self.pub_joint_vel = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )

        # --- Timers ---
        # 1. Timer d'initialisation (1 Hz)
        self.init_timer = self.create_timer(1.0, self.initialization_timer_callback)
        
        # 2. Timer de contrôle (100 Hz) - C'est lui qui publie en continu !
        self.control_timer = self.create_timer(0.01, self.control_loop)
        
        self.current_q = None
        self.get_logger().info("Nœud démarré. En attente de configuration...")

    def robot_description_callback(self, msg):
        if not self.urdf_xml:
            self.get_logger().info("URDF reçu via topic.")
            self.urdf_xml = msg.data

    def initialization_timer_callback(self):
        if self.kdl_initialized:
            return

        if not self.urdf_xml:
            self.get_logger().warn("En attente du robot description...", throttle_duration_sec=2.0)
            return

        success = self.init_kdl_from_xml(self.urdf_xml)
        if success:
            self.get_logger().info(">>> KDL INITIALISÉ AVEC SUCCÈS ! <<<")
            self.kdl_initialized = True
            self.init_timer.cancel()

    def init_kdl_from_xml(self, xml_string):
        try:
            robot_urdf = URDF.from_xml_string(xml_string)
            ok, self.kdl_tree = urdf_to_kdl(robot_urdf)        
            if not ok: return False

            found_chain = False
            for base in self.possible_base_links:
                try:
                    chain = self.kdl_tree.getChain(base, self.end_effector_link)
                    if chain.getNrOfJoints() > 0:
                        self.kdl_chain = chain
                        self.num_joints = chain.getNrOfJoints()
                        self.get_logger().info(f"Chaîne: {base} -> {self.end_effector_link} ({self.num_joints} joints)")
                        found_chain = True
                        break
                except Exception: pass
            
            if not found_chain: return False

            # Récupérer les noms des joints dans l'ordre de la chaîne KDL
            self.joint_names = []
            for i in range(self.num_joints):
                segment = self.kdl_chain.getSegment(i)
                self.joint_names.append(segment.getJoint().getName())
            
            self.get_logger().info(f"Ordre des joints KDL: {self.joint_names}")

            self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
            self.current_q = PyKDL.JntArray(self.num_joints)
            return True

        except Exception as e:
            self.get_logger().error(f"Erreur init KDL: {e}")
            return False
    
    def joint_state_callback(self, msg):
        # On stocke simplement les positions dans un dictionnaire pour accès rapide par nom
        for name, pos in zip(msg.name, msg.position):
            self.current_joint_positions[name] = pos

    def cmd_vel_callback(self, msg):
        # On met à jour la consigne et le timestamp
        v_linear = PyKDL.Vector(msg.linear.x, msg.linear.y, msg.linear.z)
        v_angular = PyKDL.Vector(msg.angular.x, msg.angular.y, msg.angular.z)
        self.target_twist = PyKDL.Twist(v_linear, v_angular)
        self.last_cmd_time = time.time()

    def control_loop(self):
        """Boucle principale exécutée à 100Hz"""
        if not self.kdl_initialized or not self.joint_names:
            return

        # 1. Mise à jour de current_q (Joint positions)
        # On s'assure d'avoir reçu les états de tous les joints nécessaires
        try:
            for i, name in enumerate(self.joint_names):
                if name in self.current_joint_positions:
                    self.current_q[i] = self.current_joint_positions[name]
                else:
                    # Si on n'a pas encore l'état d'un joint, on attend
                    return 
        except Exception:
            return

        # 2. Watchdog de sécurité
        # Si pas de commande depuis X secondes, on force la vitesse à 0
        if (time.time() - self.last_cmd_time) > self.cmd_timeout:
            active_twist = PyKDL.Twist() # Vitesse nulle
        else:
            active_twist = self.target_twist

        # 3. Calcul Cinématique Inverse (IK)
        q_dot_out = PyKDL.JntArray(self.num_joints)
        ret = self.ik_vel_solver.CartToJnt(self.current_q, active_twist, q_dot_out)

        # 4. Publication
        if ret >= 0:
            out_msg = Float64MultiArray()
            # On extrait les données du JntArray
            out_msg.data = [q_dot_out[i] for i in range(self.num_joints)]
            self.pub_joint_vel.publish(out_msg)

def main(args=None):
    rclpy.init(args=args)
    node = CartesianToJointVelocity()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()