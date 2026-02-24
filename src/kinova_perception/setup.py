# =============================================================================
# kinova_perception — Color-based cube detection and 3D pose estimation
# for the Kinova Gen3 Lite pick-and-place system.
# =============================================================================
from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'kinova_perception'

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
    description='Color-based cube detection and 3D pose estimation for Kinova pick-and-place',
    license='BSD-3-Clause',
    entry_points={
        'console_scripts': [
            'color_detector_node = kinova_perception.color_detector_node:main',
        ],
    },
)
