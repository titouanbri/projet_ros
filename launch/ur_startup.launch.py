from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # Déclaration de l'argument 'mode' pour choisir entre 'simu' et 'reel'
    mode_arg = DeclareLaunchArgument(
        'mode',
        default_value='simu',
        description="Mode de lancement : 'simu' pour Gazebo ou 'reel' pour le robot physique"
    )

    # Récupération de la valeur de l'argument
    mode = LaunchConfiguration('mode')

    # --- Configuration pour le mode SIMULATION ---
    # Commande équivalente : ros2 launch ur_simulation_gazebo ur_sim_moveit.launch.py ur_type:=ur3
    sim_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("ur_simulation_gazebo"), '/launch/ur_sim_moveit.launch.py'
        ]),
        launch_arguments={'ur_type': 'ur3'}.items(),
        # Ce bloc ne s'active que si mode == 'simu'
        condition=IfCondition(PythonExpression(["'", mode, "' == 'simu'"]))
    )

    # --- Configuration pour le mode RÉEL ---
    # Commande équivalente : ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur3 ...
    real_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare("ur_robot_driver"), '/launch/ur_control.launch.py'
        ]),
        launch_arguments={
            'ur_type': 'ur3',
            'robot_ip': '192.168.1.102',
            'launch_rviz': 'true',
        }.items(),
        # Ce bloc ne s'active que si mode == 'reel'
        condition=IfCondition(PythonExpression(["'", mode, "' == 'reel'"]))
    )

    # Commande équivalente : ros2 run projet_ros ivk
    ivk_node = Node(
        package='projet_ros',
        executable='ivk',
        name='ivk_node',
        output='screen',
        # Ce nœud ne se lance que si mode == 'reel'
        condition=IfCondition(PythonExpression(["'", mode, "' == 'reel'"]))
    )

    # --- AJOUT DEMANDÉ : RQT Controller Manager ---
    # Commande équivalente : ros2 run rqt_controller_manager rqt_controller_manager
    rqt_node = Node(
        package='rqt_controller_manager',
        executable='rqt_controller_manager',
        name='rqt_controller_manager',
        output='screen',
        # Ce nœud ne se lance que si mode == 'reel'
        condition=IfCondition(PythonExpression(["'", mode, "' == 'reel'"]))
    )

    # --- AJOUT : TF Statique de la Caméra ---
    # Publie la transformation entre tool0 (flasque du robot) et camera_link
    camera_tf_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_static_tf',
        arguments=[
            '--x', '0.05',  # <--- REMPLACEZ PAR VOTRE MESURE X (mètres)
            '--y', '0.0',   # <--- REMPLACEZ PAR VOTRE MESURE Y
            '--z', '0.03',  # <--- REMPLACEZ PAR VOTRE MESURE Z
            '--yaw', '0.0',     # <--- ROTATION Z (radians)
            '--pitch', '0.0',   # <--- ROTATION Y (radians)
            '--roll', '0.0',    # <--- ROTATION X (radians)
            '--frame-id', 'tool0',         # Parent : bout du bras UR3
            '--child-frame-id', 'camera_link'  # Enfant : votre caméra
        ],
        output='screen'
        # Pas de condition ici : actif en simu ET en réel.
    )

    return LaunchDescription([
        mode_arg,
        sim_launch,
        real_launch,
        ivk_node,
        rqt_node,       # Ajouté ici
        camera_tf_node 
    ])