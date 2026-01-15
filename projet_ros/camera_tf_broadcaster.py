import math
import sys

import rclpy
from rclpy.node import Node
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster
from geometry_msgs.msg import TransformStamped

def quaternion_from_euler(ai, aj, ak):
    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    q = [0, 0, 0, 0]
    q[0] = cj*sc - sj*cs # x
    q[1] = cj*ss + sj*cc # y
    q[2] = cj*cs - sj*sc # z
    q[3] = cj*cc + sj*ss # w (real part)
    return q

class CameraStaticTF(Node):

    def __init__(self):
        super().__init__('camera_static_tf_node')

        self.tf_static_broadcaster = StaticTransformBroadcaster(self)

        # Déclaration des paramètres (avec valeurs par défaut)
        # Tu pourras les modifier via le launch file ou un yaml
        self.declare_parameter('x', -0.05)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.01)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('parent_frame', 'tool0')
        self.declare_parameter('child_frame', 'camera_link')

        self.publish_transform()

    def publish_transform(self):
        t = TransformStamped()

        # Timestamp courant
        t.header.stamp = self.get_clock().now().to_msg()
        
        # Récupération des paramètres
        t.header.frame_id = self.get_parameter('parent_frame').get_parameter_value().string_value
        t.child_frame_id = self.get_parameter('child_frame').get_parameter_value().string_value
        
        x = self.get_parameter('x').get_parameter_value().double_value
        y = self.get_parameter('y').get_parameter_value().double_value
        z = self.get_parameter('z').get_parameter_value().double_value
        
        roll = self.get_parameter('roll').get_parameter_value().double_value
        pitch = self.get_parameter('pitch').get_parameter_value().double_value
        yaw = self.get_parameter('yaw').get_parameter_value().double_value

        # Remplissage de la translation
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = z

        # Conversion Euler (Roll/Pitch/Yaw) vers Quaternion
        # Note: L'ordre standard ROS est souvent sxyz, ici on applique une conversion standard
        q = quaternion_from_euler(roll, pitch, yaw)
        
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # Envoi de la transformation
        self.tf_static_broadcaster.sendTransform(t)
        self.get_logger().info(f'Static TF publiée : {t.header.frame_id} -> {t.child_frame_id}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraStaticTF()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()