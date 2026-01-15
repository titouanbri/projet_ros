from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    
    # 1. Configuration pour la caméra Realsense
    # Commande équivalente : ros2 launch realsense2_camera rs_launch.py
    realsense_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("realsense2_camera"), '/launch/rs_launch.py'
        ]),
        # Vous pouvez ajouter des arguments ici si nécessaire
        # launch_arguments={'align_depth.enable': 'true'}.items(),
    )

    # 2. Lancement du noeud de détection Aruco
    # Commande équivalente : ros2 run projet_ros aruko_detection
    aruco_node = Node(
        package='projet_ros',
        executable='aruko_detection',
        name='aruko_detection',
        output='screen'
    )

    # 3. Lancement du broadcaster TF pour la caméra
    # Commande équivalente : ros2 run projet_ros camera_tf_broadcaster
    tf_broadcaster_node = Node(
        package='projet_ros',          # Nom exact du package
        executable='camera_tf_broadcaster', # Nom de l'exécutable
        name='camera_tf_broadcaster',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        aruco_node,
        tf_broadcaster_node
    ])