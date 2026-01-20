from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'projet_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='titouan',
    maintainer_email='titouan.briancon@sigma-clermont.fr',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'puck_detector = projet_ros.puck_detector:main',
            'video_feed_gui = projet_ros.video_feed:main',
            'webcam_publisher = projet_ros.webcam_publisher:main',
            'aruko_detection = projet_ros.aruko_detection:main',
            'cart_vel_panel = projet_ros.cart_vel_panel:main',
            'ivk = projet_ros.ivk:main',
            'pbvs_node = projet_ros.pbvs_node:main',
            'camera_tf_broadcaster = projet_ros.camera_tf_broadcaster:main',
            'triangulate_dlt = projet_ros.triangulate_dlt:main',

        ],
    },
)
