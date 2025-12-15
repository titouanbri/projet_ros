import os
import yaml
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
# AJOUT : On importe TimerAction pour gérer le délai
from launch.actions import ExecuteProcess, TimerAction 
import xacro

def load_file(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return file.read()
    except EnvironmentError:
        return None

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)
    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:
        return None

def generate_launch_description():
    # 1. Récupération de la description du robot (URDF & SRDF)
    robot_description_config = xacro.process_file(
        os.path.join(
            get_package_share_directory('ur_description'),
            'urdf',
            'ur.urdf.xacro'
        ),
        mappings={'name': 'ur', 'ur_type': 'ur3e'}
    )
    robot_description = {'robot_description': robot_description_config.toxml()}

    srdf_file_path = os.path.join(
        get_package_share_directory('ur_moveit_config'),
        'srdf',
        'ur.srdf.xacro'
    )
    
    robot_description_semantic_config = xacro.process_file(
        srdf_file_path,
        mappings={'name': 'ur', 'prefix': ''} 
    ).toxml()

    robot_description_semantic = {'robot_description_semantic': robot_description_semantic_config}

    # 2. Chargement de TON fichier de config Servo
    servo_yaml = load_yaml('projet_ros', 'config/ur_servo_config.yaml')
    servo_params = {'moveit_servo': servo_yaml}

    # 3. Démarrage du nœud Servo
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        parameters=[
            servo_params,
            robot_description,
            robot_description_semantic,
            load_yaml('ur_moveit_config', 'config/kinematics.yaml')
        ],
        output='screen',
    )

    # --- NOUVEAU BLOC : Appel automatique du service ---
    
    # On définit la commande shell équivalente à ce que tu tapes dans le terminal
    start_servo_command = ExecuteProcess(
        cmd=['ros2', 'service', 'call', '/servo_node/start_servo', 'std_srvs/srv/Trigger', '{}'],
        output='screen'
    )

    # On demande à ROS d'attendre 5 secondes avant de lancer cette commande
    # Cela laisse le temps à 'servo_node' de charger et de créer le service.
    delayed_start_servo = TimerAction(
        period=2.0, 
        actions=[start_servo_command]
    )
    # ---------------------------------------------------

    return LaunchDescription([
        servo_node,
        delayed_start_servo # Ne pas oublier d'ajouter l'action ici
    ])