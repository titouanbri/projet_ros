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


        self.init_dlt=True   #defiine if we need to init the dlt

        self.lmbda = 5      # Gain proportionnel (lambda)  1
        self.dist_target = 0.1  # Distance désirée entre le marker et la cam   0.1
        
        self.target_frame = 'puck_link'
        # self.target_frame = 'aruco_0'
        self.camera_frame = 'camera_color_optical_frame'
        self.tool_frame = 'tool0'
        
        self.MAX_LIN_VEL = 0.01 # m/s   0.05
        self.MAX_ANG_VEL = 0.05  # rad/s     0.5

        #pose désirée devant le marqueur        
        rot_target = R.from_euler('x', -180, degrees=True).as_matrix()
        pos_target = np.array([0.0, 0.0, self.dist_target])
        
        self.T_des = np.eye(4)
        self.T_des[:3, :3] = rot_target
        self.T_des[:3, 3] = pos_target

        # SETUP TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher vitesse
        self.vel_pub = self.create_publisher(Twist, '/ee_velocity_cmd', 10)
        
        # Timer de contrôle 
        self.timer = self.create_timer(0.1, self.control_loop)
        
        self.get_logger().info("node launched")

    def transform_to_matrix(self, t_stamped):
        #transform a TF in a matrix 4x4
        t = t_stamped.transform.translation
        r = t_stamped.transform.rotation
        
        mat = np.eye(4)
        mat[:3, :3] = R.from_quat([r.x, r.y, r.z, r.w]).as_matrix()
        mat[:3, 3] = [t.x, t.y, t.z]
        return mat

    def limit_velocity(self, v, max_val):
        
        norm = np.linalg.norm(v)
        if norm > max_val:
            return v * (max_val / norm)  #keep the direction
        return v

    def control_loop(self):       
        try:
            # marker/cam TF
            t_cam_marker = self.tf_buffer.lookup_transform(
                self.camera_frame,      
                self.target_frame,      #marker
                rclpy.time.Time()
            )
            # current time
            now = self.get_clock().now()
            # timestamp de la TF
            tf_time = rclpy.time.Time.from_msg(t_cam_marker.header.stamp)
            age = (now - tf_time).nanoseconds / 1e9
            
            if age > 0.5:
                # if too old, stop
                self.vel_pub.publish(Twist()) # STOP
                return
        except TransformException as ex:
            self.vel_pub.publish(Twist()) # Stop
            return

        # Convert to matrix 
        T_curr = self.transform_to_matrix(t_cam_marker)
         
        # error in camera frame (mezouar TP1)
        X = self.T_des @ np.linalg.inv(T_curr)

        R_mat = X[:3, :3]
        t_vec = X[:3, 3]

        r_obj = R.from_matrix(R_mat)    #intermediate rotation object
        rot_vec = r_obj.as_rotvec()   #rotation vector (angle-axis)

        # Loi de commande PBVS dans repère caméra
        v_cam = -self.lmbda * (R_mat.T @ t_vec)
        w_cam = -self.lmbda * rot_vec

        # passage de Caméra -> Tool via TF
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

        v_cam_in_tool = R_tc @ v_cam
        w_cam_in_tool = R_tc @ w_cam
        
        v_tool = v_cam_in_tool + np.cross(P_tc, w_cam_in_tool)
        w_tool = w_cam_in_tool

        # limitation des vitesses
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