import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

def generate_launch_description():
    # 1. Configuration pour lancer realsense2_camera
    # Cela correspond à : ros2 launch realsense2_camera rs_launch.py
    realsense_dir = get_package_share_directory('realsense2_camera')
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(realsense_dir, 'launch', 'rs_launch.py')
        )
        # Vous pouvez ajouter des arguments ici si nécessaire, par exemple :
        # launch_arguments={'align_depth.enable': 'true'}.items()
    )

    # Correspond à : ros2 run projet_ros puck_detector
    puck_detector_node = Node(
        package='projet_ros',
        executable='puck_detector',
        name='puck_detector',
        output='screen'
        # Note : Si votre script charge un modèle avec un chemin relatif ("models/..."),
        # assurez-vous que le fichier est accessible depuis le dossier d'exécution
        # ou utilisez un chemin absolu dans le script Python.
    )


    # Correspond à : ros2 run projet_ros camera_tf_broadcaster
    tf_broadcaster_node = Node(
        package='projet_ros',
        executable='camera_tf_broadcaster',
        name='camera_tf_broadcaster',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        puck_detector_node,
        tf_broadcaster_node
    ])