from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    ur_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("ur_robot_driver"), '/launch/ur_control.launch.py'
        ]),
        launch_arguments={
            'ur_type': 'ur3e',
            'robot_ip': '192.168.1.102',
            'initial_joint_controller': 'forward_velocity_controller',
            'launch_rviz': 'true',  # Optionnel : mettez 'false' si vous ne voulez pas Rviz
        }.items()
    )

    ivk_node = Node(
        package='projet_ros',
        executable='ivk',
        name='ivk_node',
        output='screen'
    )

    return LaunchDescription([
        ur_driver_launch,
        ivk_node
    ])