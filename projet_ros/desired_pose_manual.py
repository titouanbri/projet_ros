import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster
import numpy as np
import math

def quaternion_from_rpy(roll, pitch, yaw):
    """Convert roll, pitch, yaw to quaternion (x,y,z,w)."""
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw

class DesiredPosePublisher(Node):
    def __init__(self):
        super().__init__('desired_pose_publisher')

        self.broadcaster = TransformBroadcaster(self)

        # Manually define pose in base_link frame
        self.translation = np.array([0.3, 0.3, 0.3])  # x, y, z in meters
        self.rotation_rpy = np.array([-np.pi/2, 0.0, 0.0]) # roll, pitch, yaw in radians

        self.timer = self.create_timer(0.01, self.publish_tf)

    def publish_tf(self):
        qx, qy, qz, qw = quaternion_from_rpy(*self.rotation_rpy)

        tf_msg = TransformStamped()
        tf_msg.header.stamp = self.get_clock().now().to_msg()
        tf_msg.header.frame_id = 'base_link'
        tf_msg.child_frame_id = 'desired_pose'

        tf_msg.transform.translation.x = float(self.translation[0])
        tf_msg.transform.translation.y = float(self.translation[1])
        tf_msg.transform.translation.z = float(self.translation[2])
        tf_msg.transform.rotation.x = qx
        tf_msg.transform.rotation.y = qy
        tf_msg.transform.rotation.z = qz
        tf_msg.transform.rotation.w = qw

        self.broadcaster.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = DesiredPosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
