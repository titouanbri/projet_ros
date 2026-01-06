import rclpy
from rclpy.node import Node
import numpy as np
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
import message_filters
from rclpy.qos import QoSProfile

class StereoDLTNode(Node):
    def __init__(self):
        super().__init__('stereo_dlt_node')

        # ---------------------------------------------------------
        # 1. Paramètres Intrinsèques (Hardcodés comme dans SimulatedCameraTP.m)
        # ---------------------------------------------------------
        # Dans un vrai robot, on utiliserait le topic camera_info
        self.fx = 800.0
        self.fy = 800.0
        self.u0 = 240.0
        self.v0 = 320.0
        
        self.K = np.array([
            [self.fx, 0,      self.u0],
            [0,       self.fy, self.v0],
            [0,       0,      1.0]
        ])

        # ---------------------------------------------------------
        # 2. Configuration TF (Pour remplacer X_AR et X_CR)
        # ---------------------------------------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        
        # Noms des frames (A adapter selon votre setup)
        self.world_frame = "world"    # Le repère de référence (Ref)
        self.cam1_frame = "camera_1"  # Repère optique Caméra 1
        self.cam2_frame = "camera_2"  # Repère optique Caméra 2

        # ---------------------------------------------------------
        # 3. Subscribers (Synchronisés)
        # ---------------------------------------------------------
        # On attend que les deux mesures arrivent en même temps (approx)
        qos = QoSProfile(depth=10)
        self.sub_cam1 = message_filters.Subscriber(self, PointStamped, '/camera1/target_2d')
        self.sub_cam2 = message_filters.Subscriber(self, PointStamped, '/camera2/target_2d')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.sub_cam1, self.sub_cam2], queue_size=10, slop=0.1
        )
        self.ts.registerCallback(self.callback)

        # ---------------------------------------------------------
        # 4. Publisher
        # ---------------------------------------------------------
        self.pub_3d = self.create_publisher(PointStamped, '/stereo/target_3d', 10)

        self.get_logger().info("Nœud Stereo DLT démarré. En attente de points...")

    def get_projection_matrix(self, target_frame, source_frame):
        """
        Calcule P = K * [R|t]
        Equivalent dans votre code MATLAB à : K * inv(X_AR)
        """
        try:
            # On cherche la transfo de World -> Camera
            # C'est l'inverse de la pose de la caméra dans le monde (X_AR)
            t = self.tf_buffer.lookup_transform(
                target_frame,      # Camera frame
                source_frame,      # World frame
                rclpy.time.Time()
            )
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(f'Pas de transformation trouvée: {ex}')
            return None

        # Conversion Quaternion -> Matrice de Rotation
        q = [t.transform.rotation.x, t.transform.rotation.y, 
             t.transform.rotation.z, t.transform.rotation.w]
        
        # Conversion simple quaternion vers matrice 3x3 (formule standard)
        x, y, z, w = q
        R = np.array([
            [1 - 2*y*y - 2*z*z,  2*x*y - 2*z*w,      2*x*z + 2*y*w],
            [2*x*y + 2*z*w,      1 - 2*x*x - 2*z*z,  2*y*z - 2*x*w],
            [2*x*z - 2*y*w,      2*y*z + 2*x*w,      1 - 2*x*x - 2*y*y]
        ])

        # Vecteur translation
        T = np.array([
            [t.transform.translation.x],
            [t.transform.translation.y],
            [t.transform.translation.z]
        ])

        # Matrice Extrinsèque [R|t] (3x4)
        Extrinsics = np.hstack((R, T))

        # Matrice de Projection P = K * [R|t]
        P = self.K @ Extrinsics
        return P

    def triangulate_DLT_algo(self, m1, P1, m2, P2):
        """
        Traduction exacte de votre fonction triangulate_DLT.m
        """
        # Construction des matrices Skew
        # m1 = [u, v, 1]
        
        m1_skew = np.array([
            [0,      -m1[2],  m1[1]],
            [m1[2],   0,     -m1[0]],
            [-m1[1],  m1[0],  0]
        ])
        
        m2_skew = np.array([
            [0,      -m2[2],  m2[1]],
            [m2[2],   0,     -m2[0]],
            [-m2[1],  m2[0],  0]
        ])

        # Construction de A (Ax=0)
        A1 = m1_skew @ P1
        A2 = m2_skew @ P2
        A = np.vstack((A1, A2))

        # SVD pour trouver min(Ax)
        # numpy svd renvoie U, S, Vt (Vt est transposée de V)
        U, S, Vt = np.linalg.svd(A)
        
        # La solution est la dernière colonne de V (ou dernière ligne de Vt)
        # Equivalent MATLAB: V(:, end)
        M_homog = Vt[-1] 

        return M_homog

    def callback(self, msg1, msg2):
        # 1. Récupérer les matrices de projection P1 et P2
        # P = K * Transformation(World -> Cam)
        P1 = self.get_projection_matrix(self.cam1_frame, self.world_frame)
        P2 = self.get_projection_matrix(self.cam2_frame, self.world_frame)

        if P1 is None or P2 is None:
            return

        # 2. Préparer les points de mesure homogènes [u, v, 1]
        # Dans ROS, msg.point.x est souvent utilisé pour u et y pour v en 2D
        uv1 = np.array([msg1.point.x, msg1.point.y, 1.0])
        uv2 = np.array([msg2.point.x, msg2.point.y, 1.0])

        # 3. Appeler l'algo DLT
        X_homog = self.triangulate_DLT_algo(uv1, P1, uv2, P2)

        # 4. Normalisation Homogène -> Euclidien
        # X = X_homog(1:3) / X_homog(4)
        if abs(X_homog[3]) > 1e-6:
            X_euclidian = X_homog[:3] / X_homog[3]
        else:
            self.get_logger().warn("Point à l'infini détecté")
            return

        # 5. Publication
        res_msg = PointStamped()
        res_msg.header.stamp = msg1.header.stamp
        res_msg.header.frame_id = self.world_frame
        res_msg.point.x = X_euclidian[0]
        res_msg.point.y = X_euclidian[1]
        res_msg.point.z = X_euclidian[2]

        self.pub_3d.publish(res_msg)
        
        # Log pour debug
        self.get_logger().info(f"Triangulé: [{X_euclidian[0]:.2f}, {X_euclidian[1]:.2f}, {X_euclidian[2]:.2f}]")

def main(args=None):
    rclpy.init(args=args)
    node = StereoDLTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()