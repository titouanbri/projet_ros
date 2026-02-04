from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    ur_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ur_robot_driver'),
                'launch',
                'ur_control.launch.py'
            )
        ),
        launch_arguments={
            'ur_type': 'ur3e',
            'robot_ip': '192.168.1.102',
            'initial_joint_controller': 'forward_velocity_controller'
        }.items()
    )

    realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('realsense2_camera'),
                'launch',
                'rs_launch.py'
            )
        ),
    )

    ivk_node = Node(
        package='projet_ros',
        executable='ivk',
        name='ivk'
    )

    control_panel_node = Node(
        package='projet_ros',
        executable='control_panel',
        name='control_panel'
    )

    home_node = Node(
        package='projet_ros',
        executable='home',
        name='home'
    )

    return LaunchDescription([
        ur_launch,
        ivk_node,
        control_panel_node,
        home_node,
        realsense
    ])
