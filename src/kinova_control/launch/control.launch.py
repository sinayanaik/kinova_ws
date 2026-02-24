#!/usr/bin/env python3
"""
control.launch.py
Launches the pick_and_place_node with its parameter configuration and
the robot_description for the IK solver.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, Command, FindExecutable
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    control_share = get_package_share_directory('kinova_control')
    kinova_desc_share = get_package_share_directory('kinova_description')
    params_file = os.path.join(control_share, 'config', 'control_params.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time')

    # Generate robot description for Pinocchio IK solver
    xacro_file = os.path.join(kinova_desc_share, 'urdf', 'gen3_lite_environment.urdf.xacro')
    robot_description = ParameterValue(Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' sim_gazebo:=true',
        ' use_fake_hardware:=false',
        ' simulation_controllers:=',
        os.path.join(kinova_desc_share, 'config', 'controllers.yaml'),
    ]), value_type=str)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),

        Node(
            package='kinova_control',
            executable='pick_and_place_node',
            name='pick_and_place_node',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'robot_description': robot_description,
                },
            ],
        ),
    ])
