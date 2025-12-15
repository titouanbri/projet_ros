import sys
import rclpy
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

        # Liste de tentatives pour la base si la première échoue
        self.possible_base_links = ['base_link_inertia', 'base_link', 'base', 'world']
        self.end_effector_link = 'tool0' 
        
        self.kdl_tree = None
        self.kdl_chain = None
        self.ik_vel_solver = None
        self.num_joints = 0
        self.urdf_xml = None # Stockage de l'URDF
        self.kdl_initialized = False

        # --- Initialisation des Paramètres ---
        self.declare_parameter('robot_description_str', '') 
        param_desc = self.get_parameter('robot_description_str').value
        if param_desc:
            self.urdf_xml = param_desc

        # --- Timer de réessai (1 Hz) ---
        # C'est ici que la magie opère : on vérifie chaque seconde si on peut initialiser
        self.init_timer = self.create_timer(1.0, self.initialization_timer_callback)

        # --- Subscribers & Publishers ---
        self.create_subscription(
            String,
            '/robot_description',
            self.robot_description_callback,
            qos_profile=rclpy.qos.QoSProfile(depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        )
        
        self.sub_joint_states = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        self.sub_cmd_vel = self.create_subscription(
            Twist,
            '/cmd_vel_input',
            self.cmd_vel_callback,
            10
        )

        self.pub_joint_vel = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )
        
        self.current_q = None
        self.get_logger().info("Nœud démarré. En attente de configuration...")

    def robot_description_callback(self, msg):
        """Stocke l'URDF reçu pour que le Timer l'utilise"""
        if not self.urdf_xml:
            self.get_logger().info("URDF reçu via topic ! Stockage en mémoire.")
            self.urdf_xml = msg.data

    def initialization_timer_callback(self):
        """Tente d'initialiser KDL en boucle jusqu'à succès"""
        if self.kdl_initialized:
            # Si c'est déjà bon, on ne fait rien (ou on pourrait détruire le timer)
            return

        if not self.urdf_xml:
            self.get_logger().warn("En attente du robot description...", throttle_duration_sec=2.0)
            return

        # Tentative d'initialisation
        success = self.init_kdl_from_xml(self.urdf_xml)
        if success:
            self.get_logger().info(">>> KDL INITIALISÉ AVEC SUCCÈS ! <<<")
            self.kdl_initialized = True
            # Optionnel : Arrêter le timer pour économiser des ressources
            self.init_timer.cancel()
        else:
            self.get_logger().error("Échec initialisation KDL (Chaîne vide). Nouvelle tentative dans 1s...")

    def init_kdl_from_xml(self, xml_string):
        try:
            robot_urdf = URDF.from_xml_string(xml_string)
            ok, self.kdl_tree = urdf_to_kdl(robot_urdf)        
            if not ok:
                return False

            # On essaie de trouver une chaîne valide parmi les bases possibles
            found_chain = False
            
            for base in self.possible_base_links:
                try:
                    chain = self.kdl_tree.getChain(base, self.end_effector_link)
                    n_joints = chain.getNrOfJoints()
                    
                    if n_joints > 0:
                        self.kdl_chain = chain
                        self.num_joints = n_joints
                        self.get_logger().info(f"Chaîne trouvée entre '{base}' et '{self.end_effector_link}' avec {n_joints} joints.")
                        found_chain = True
                        break # Sort de la boucle for
                    else:
                        self.get_logger().debug(f"Chaîne vide pour base '{base}'")
                except Exception:
                    pass
            
            if not found_chain:
                self.get_logger().warn(f"Impossible de trouver une chaîne vers '{self.end_effector_link}' (Joints=0).")
                self.get_logger().warn(f"Links testés comme base: {self.possible_base_links}")
                return False

            # Initialisation du solver et des vecteurs
            self.ik_vel_solver = PyKDL.ChainIkSolverVel_pinv(self.kdl_chain)
            self.current_q = PyKDL.JntArray(self.num_joints)
            
            return True

        except Exception as e:
            self.get_logger().error(f"Exception lors du parsing KDL: {e}")
            return False
    
    def joint_state_callback(self, msg):
        if not self.kdl_initialized or self.current_q is None:
            return
        
        # Note simplifiée : on suppose que l'ordre correspond. 
        # Pour une vraie robustesse, il faut mapper joint_names -> indices
        if len(msg.position) >= self.num_joints:
            for i in range(self.num_joints):
                self.current_q[i] = msg.position[i]

    def cmd_vel_callback(self, msg):
        if not self.kdl_initialized:
            return

        v_linear = PyKDL.Vector(msg.linear.x, msg.linear.y, msg.linear.z)
        v_angular = PyKDL.Vector(msg.angular.x, msg.angular.y, msg.angular.z)
        kdl_twist = PyKDL.Twist(v_linear, v_angular)

        q_dot_out = PyKDL.JntArray(self.num_joints)
        ret = self.ik_vel_solver.CartToJnt(self.current_q, kdl_twist, q_dot_out)

        if ret >= 0:
            out_msg = Float64MultiArray()
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
        # Correction pour éviter l'erreur de shutdown double
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()