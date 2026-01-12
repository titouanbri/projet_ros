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
        # Vous pouvez ajouter des arguments ici si nécessaire, par exemple :
        # launch_arguments={'align_depth.enable': 'true'}.items(),
    )

    # 2. Lancement du noeud de détection Aruco
    # Commande équivalente : ros2 run projet_ros aruko_detection
    # (Note : le nom du package est 'projet_ros' selon votre setup.py)
    aruco_node = Node(
        package='projet_ros',          # Nom exact du package
        executable='aruko_detection',  # Nom défini dans entry_points console_scripts
        name='aruko_detection',
        output='screen'
    )

    return LaunchDescription([
        realsense_launch,
        aruco_node
    ])