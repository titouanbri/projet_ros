from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # Lance le nœud pbvs_node
        Node(
            package='projet_ros',
            executable='pbvs_node',
            name='pbvs_node',
            output='screen'
        ),
        # Lance le nœud pose_control (déclaré comme cam_pose_controller dans setup.py)
        Node(
            package='projet_ros',
            executable='cam_pose_controller',
            name='pose_control',
            output='screen'
        ),
        # Lance le nœud supervisor_node
        Node(
            package='projet_ros',
            executable='supervisor_node',
            name='supervisor_node',
            output='screen'
        ),
    ])