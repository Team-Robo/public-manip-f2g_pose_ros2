from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'f2g_pose_ros2'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(include=['f2g_pose_ros2*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    scripts=['scripts/object_perception_node'],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nanyang Technological University Team Robo',
    maintainer_email='teamrobontu@gmail.com',
    author='Hu Ming Jie',
    description='F2G-Pose ROS 2 wrapper for single-view 3D shape completion and 6D pose estimation',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'f2g_pose_node = f2g_pose_ros2.f2g_pose_node:main',
        ],
    },
)
