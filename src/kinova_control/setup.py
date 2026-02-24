# =============================================================================
# kinova_control — IK solver, motion execution, gripper control, grasp
# verification, and state machine for the Kinova Gen3 Lite pick-and-place.
# =============================================================================
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kinova_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='IK, motion planning, and state machine for Kinova Gen3 Lite pick-and-place',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'pick_and_place_node = kinova_control.pick_and_place_node:main',
        ],
    },
)
