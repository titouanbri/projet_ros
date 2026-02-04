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
            'ivk = projet_ros.ivk:main',
            'home = projet_ros.home:main',
            'control_panel = projet_ros.cart_vel_panel:main',
            'calibration_routine = projet_ros.calibration_routine:main',
            'calib_cam_tf_broadcaster = projet_ros.calib_cam_tf_broadcaster:main'

        ],
    },
)
