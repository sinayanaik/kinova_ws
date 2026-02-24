#!/usr/bin/env python3
"""
perception.launch.py
Launches the color_detector_node with its parameter configuration.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    perception_share = get_package_share_directory('kinova_perception')
    params_file = os.path.join(perception_share, 'config', 'perception_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='kinova_perception',
            executable='color_detector_node',
            name='color_detector',
            output='screen',
            parameters=[params_file, {'use_sim_time': use_sim_time}],
        ),
    ])
