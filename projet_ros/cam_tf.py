#!/usr/bin/env python3
import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster, Buffer, TransformListener
from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
import tf2_geometry_msgs
import numpy as np


def quaternion_to_rotation_matrix(qx, qy, qz, qw):
    """Convert quaternion to 3x3 rotation matrix."""
    # Normalize quaternion
    norm = (qx**2 + qy**2 + qz**2 + qw**2)**0.5
    qx, qy, qz, qw = qx/norm, qy/norm, qz/norm, qw/norm
    
    # Compute rotation matrix elements
    r00 = 1 - 2*(qy**2 + qz**2)
    r01 = 2*(qx*qy - qz*qw)
    r02 = 2*(qx*qz + qy*qw)
    
    r10 = 2*(qx*qy + qz*qw)
    r11 = 1 - 2*(qx**2 + qz**2)
    r12 = 2*(qy*qz - qx*qw)
    
    r20 = 2*(qx*qz - qy*qw)
    r21 = 2*(qy*qz + qx*qw)
    r22 = 1 - 2*(qx**2 + qy**2)
    
    return np.array([
        [r00, r01, r02],
        [r10, r11, r12],
        [r20, r21, r22]
    ])

class ContinuousTF(Node):
    def __init__(self):
        super().__init__('continuous_aruco_tf')
        
        # --- fixed offsets from wrist_3_link ---
        self.x_offset = -0.05#0.0
        self.y_offset = 0.03#-0.05
        self.z_offset = 0.03
        
        self.parent_frame = 'base_link'
        self.wrist_frame = 'wrist_3_link'
        self.parent_frame = self.wrist_frame
        self.child_frame = 'camera_link'
        
        self.tf_broadcaster = TransformBroadcaster(self)
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        
        # 100 Hz
        self.timer = self.create_timer(0.01, self.publish_tf)
        self.get_logger().info(f"Broadcasting TF {self.parent_frame} -> {self.child_frame} at 100Hz")
    
    def publish_tf(self):
        try:
            # Get wrist_3_link pose in base_link
            trans = self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.wrist_frame,
                rclpy.time.Time()
            )


            q = trans.transform.rotation
            quat = [q.x, q.y, q.z, q.w]
            R_ee = quaternion_to_rotation_matrix(q.x, q.y, q.z, q.w)[:3, :3]
            z_ee = R_ee[:3, 2];y_ee = R_ee[:3, 1];x_ee = R_ee[:3, 2]

            offset = x_ee * self.x_offset + y_ee * self.y_offset + z_ee * self.z_offset


            
            # Apply offset to get camera_link position
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.parent_frame
            t.child_frame_id = self.child_frame
            
            # Add offset to wrist position
            t.transform.translation.x = self.x_offset#float(trans.transform.translation.x + offset[0])
            t.transform.translation.y = self.y_offset#float(trans.transform.translation.y + offset[1])
            t.transform.translation.z = self.z_offset#float(trans.transform.translation.z + offset[2])
            
            # Use wrist orientation
            t.transform.rotation = trans.transform.rotation
            
            self.tf_broadcaster.sendTransform(t)
            
        except (LookupException, ConnectivityException, ExtrapolationException) as e:
            self.get_logger().warn(f'TF lookup failed: {str(e)}', throttle_duration_sec=5.0)
        except Exception as e:
            self.get_logger().error(f'Failed to publish transform: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()