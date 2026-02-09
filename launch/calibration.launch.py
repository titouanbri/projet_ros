from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    calibration_routine = Node(
        package='projet_ros',
        executable='calibration_routine',
        name='calibration_routine'
    )

    calibrated_cam_broadcaster = Node(
        package='projet_ros',
        executable='calib_cam_tf_broadcaster',
        name='calib_cam_tf_broadcaster'
    )

    pose_control = Node(
        package='projet_ros',
        executable='pose_control',
        name='pose_control'
    )

    return LaunchDescription([
        calibration_routine,
        calibrated_cam_broadcaster,
        pose_control
    ])
