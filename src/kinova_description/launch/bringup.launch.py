#!/usr/bin/env python3
# =============================================================================
# bringup.launch.py
# Master launch file for the Kinova Gen3 Lite pick-and-place system.
# Spawns: Gazebo world → robot → controllers → cameras (bridge) →
#         perception node → control node
# =============================================================================
import os
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    ExecuteProcess,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    Command,
    FindExecutable,
    LaunchConfiguration,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ----- Package paths -----
    kinova_desc_share = get_package_share_directory('kinova_description')
    perception_share = get_package_share_directory('kinova_perception')
    control_share = get_package_share_directory('kinova_control')

    # ----- Launch arguments -----
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_file = LaunchConfiguration('world_file')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    declare_world_file = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(kinova_desc_share, 'worlds', 'pick_and_place.sdf'),
        description='Path to the Gazebo world SDF file'
    )

    # ----- Robot description (xacro → URDF) -----
    xacro_file = os.path.join(kinova_desc_share, 'urdf', 'gen3_lite_environment.urdf.xacro')
    robot_description_content = Command([
        FindExecutable(name='xacro'), ' ', xacro_file,
        ' sim_gazebo:=true',
        ' use_fake_hardware:=false',
        ' simulation_controllers:=',
        os.path.join(kinova_desc_share, 'config', 'controllers.yaml'),
    ])

    robot_description = {'robot_description': robot_description_content}

    # ----- Nodes -----

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    # Gazebo Sim
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py'
        ]),
        launch_arguments={
            'gz_args': ['-r -v 4 ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
    )

    # Spawn robot into Gazebo
    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=[
            '-name', 'kinova_gen3_lite',
            '-topic', 'robot_description',
            '-allow_renaming', 'true',
        ],
        output='screen',
    )

    # Controller spawners (delayed after robot spawn)
    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    spawn_arm_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    spawn_gripper_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['gripper_trajectory_controller', '--controller-manager', '/controller_manager'],
        output='screen',
    )

    # Chain controller spawns: JSB → arm → gripper
    delayed_arm_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_joint_state_broadcaster,
            on_exit=[spawn_arm_controller],
        )
    )
    delayed_gripper_controller = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=spawn_arm_controller,
            on_exit=[spawn_gripper_controller],
        )
    )

    # ros_gz_bridge — bridge Gazebo camera topics to ROS2
    ros_gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/camera_overhead/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_overhead/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/camera_side/image@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_side/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[
            ('/camera_overhead/image', '/camera_overhead/image_raw'),
            ('/camera_side/image', '/camera_side/image_raw'),
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Perception launch (delayed to let cameras initialize)
    perception_launch = TimerAction(
        period=8.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(perception_share, 'launch', 'perception.launch.py')
                ),
                launch_arguments={'use_sim_time': 'true'}.items(),
            )
        ],
    )

    # Control launch (delayed to let perception + controllers initialize)
    control_launch = TimerAction(
        period=12.0,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(control_share, 'launch', 'control.launch.py')
                ),
                launch_arguments={'use_sim_time': 'true'}.items(),
            )
        ],
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world_file,
        robot_state_publisher,
        gazebo,
        spawn_robot,
        spawn_joint_state_broadcaster,
        delayed_arm_controller,
        delayed_gripper_controller,
        ros_gz_bridge,
        perception_launch,
        control_launch,
    ])
