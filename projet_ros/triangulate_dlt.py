#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rclpy
from rclpy.node import Node
# AJOUT : TransformStamped nécessaire pour la TF
from geometry_msgs.msg import Point, PointStamped, TransformStamped
from sensor_msgs.msg import CameraInfo
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
# AJOUT : Import du Broadcaster TF
from tf2_ros import TransformBroadcaster
import numpy as np

class DLTTriangulatorNode(Node):
    def __init__(self):
        super().__init__('dlt_triangulator_node')

        # --- Paramètres ---
        self.declare_parameter('time_interval', 0.3) 
        self.declare_parameter('min_baseline', 0.001) 
        self.declare_parameter('world_frame', 'base_link')
        self.declare_parameter('camera_frame', 'camera_link')
        self.declare_parameter('buffer_size', 5) 

        self.time_interval = self.get_parameter('time_interval').value
        self.min_baseline = self.get_parameter('min_baseline').value
        self.world_frame = self.get_parameter('world_frame').value
        self.camera_frame = self.get_parameter('camera_frame').value
        self.buffer_size = self.get_parameter('buffer_size').value

        # --- Variables d'état ---
        self.camera_matrix = None 
        self.views = [] 
        
        # --- TF Setup ---
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # AJOUT : Initialisation du Broadcaster TF (comme dans aruko_detection)
        self.tf_broadcaster = TransformBroadcaster(self)

        # --- Subscribers / Publishers ---
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/camera/camera/color/camera_info', 
            self.info_callback,
            10
        )
        
        self.pixel_sub = self.create_subscription(
            Point,
            '/puck/position',
            self.pixel_callback,
            10
        )

        self.point_pub = self.create_publisher(PointStamped, '/dlt/triangulated_point', 10)

        self.get_logger().info(f"DLT N-Views Node Started. Buffer size: {self.buffer_size}")

    def info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape((3, 3))
            self.get_logger().info("Calibration caméra (K) reçue.")

    def get_projection_matrix(self, time_point):
        """ Calcule P = K * [R|t]^-1 """
        try:
            t = self.tf_buffer.lookup_transform(
                self.camera_frame, 
                self.world_frame, 
                time_point
            )
        except TransformException as ex:
            self.get_logger().warn(f'TF Error: {ex}')
            return None, None

        q = [t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w]
        R = self.quaternion_to_rotation_matrix(q)
        T = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]).reshape(3, 1)

        RT = np.hstack((R, T))
        P = self.camera_matrix @ RT
        
        cam_pos_in_world = -R.T @ T 
        
        return P, cam_pos_in_world.flatten()

    def pixel_callback(self, msg):
        if self.camera_matrix is None:
            return

        pixel_homog = np.array([msg.x, msg.y, 1.0])
        
        current_time = rclpy.time.Time() 
        P_curr, cam_pos_curr = self.get_projection_matrix(current_time)
        
        if P_curr is None:
            return

        # --- Logique de gestion du Buffer (N points) ---
        candidate_view = {
            'pixel': pixel_homog,
            'P': P_curr,
            'pos': cam_pos_curr,
            'time': self.get_clock().now()
        }

        should_add = False
        if len(self.views) == 0:
            should_add = True
        else:
            last_view = self.views[-1]
            time_diff = (self.get_clock().now() - last_view['time']).nanoseconds / 1e9
            dist_diff = np.linalg.norm(cam_pos_curr - last_view['pos'])

            if time_diff >= self.time_interval and dist_diff >= self.min_baseline:
                should_add = True

        if should_add:
            self.views.append(candidate_view)
            
            if len(self.views) > self.buffer_size:
                self.views.pop(0) 

            if len(self.views) == self.buffer_size:
                M_3d = self.triangulate_n_views(self.views)
                
                # Timestamp commun pour le message et la TF
                now_stamp = self.get_clock().now().to_msg()

                # --- 1. Publication PointStamped ---
                res_msg = PointStamped()
                res_msg.header.stamp = now_stamp
                res_msg.header.frame_id = self.world_frame
                res_msg.point.x = M_3d[0]
                res_msg.point.y = M_3d[1]
                res_msg.point.z = M_3d[2]
                self.point_pub.publish(res_msg)

                # --- 2. Envoi de la TF (AJOUT) ---
                t = TransformStamped()
                
                # Header
                t.header.stamp = now_stamp
                t.header.frame_id = self.world_frame
                t.child_frame_id = 'puck' # Nom de la frame créée
                
                # Translation (Position calculée)
                t.transform.translation.x = float(M_3d[0])
                t.transform.translation.y = float(M_3d[1])
                t.transform.translation.z = float(M_3d[2])
                
                # Rotation (Identité car DLT ne donne pas d'orientation)
                t.transform.rotation.x = 0.0
                t.transform.rotation.y = 0.0
                t.transform.rotation.z = 0.0
                t.transform.rotation.w = 1.0 # w=1 est neutre
                
                self.tf_broadcaster.sendTransform(t)
                
                self.get_logger().info(f"Triangulation -> TF 'puck' @ [{M_3d[0]:.3f}, {M_3d[1]:.3f}, {M_3d[2]:.3f}]")

    def triangulate_n_views(self, views_list):
        A_list = []

        for view in views_list:
            m = view['pixel'] 
            P = view['P']     
            
            m_skew = np.array([
                [0,      -m[2],  m[1]],
                [m[2],   0,     -m[0]],
                [-m[1],  m[0],  0]
            ])
            
            Ai = np.dot(m_skew, P)
            A_list.append(Ai)

        A = np.vstack(A_list)
        U, S, Vh = np.linalg.svd(A)
        M_homog = Vh[-1]
        M_3d = M_homog[:3] / M_homog[3]
        
        return M_3d

    def quaternion_to_rotation_matrix(self, q):
        x, y, z, w = q
        return np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])

def main(args=None):
    rclpy.init(args=args)
    node = DLTTriangulatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()