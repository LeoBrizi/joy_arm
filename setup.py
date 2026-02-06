from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'joy_arm'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Joystick-based Cartesian control of robot arm',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'joy_arm_node = joy_arm.joy_arm_node:main',
            'joy_joint_node = joy_arm.joy_joint_node:main',
        ],
    },
)
