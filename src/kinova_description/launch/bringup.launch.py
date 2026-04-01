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
from launch.conditions import IfCondition, UnlessCondition
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
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # ----- Package paths -----
    kinova_desc_share = get_package_share_directory('kinova_description')
    perception_share = get_package_share_directory('kinova_perception')
    control_share = get_package_share_directory('kinova_control')

    # ----- Launch arguments -----
    use_sim_time = LaunchConfiguration('use_sim_time')
    world_file = LaunchConfiguration('world_file')
    gui = LaunchConfiguration('gui')
    rqt = LaunchConfiguration('rqt')
    headless_rendering = LaunchConfiguration('headless_rendering')
    gz_verbosity = LaunchConfiguration('gz_verbosity')

    declare_use_sim_time = DeclareLaunchArgument(
        'use_sim_time', default_value='true',
        description='Use simulation (Gazebo) clock'
    )
    declare_world_file = DeclareLaunchArgument(
        'world_file',
        default_value=os.path.join(kinova_desc_share, 'worlds', 'pick_and_place.sdf'),
        description='Path to the Gazebo world SDF file'
    )
    declare_gui = DeclareLaunchArgument(
        'gui', default_value='true',
        description='Start Gazebo GUI as a separate optional process'
    )
    declare_rqt = DeclareLaunchArgument(
        'rqt', default_value='true',
        description='Start rqt_image_view for the processed overhead camera'
    )
    declare_headless = DeclareLaunchArgument(
        'headless_rendering', default_value='true',
        description='Run Gazebo server with headless rendering enabled'
    )
    declare_gz_verbosity = DeclareLaunchArgument(
        'gz_verbosity', default_value='3',
        description='Gazebo verbosity level (0-4)'
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

    robot_description = {
        'robot_description': ParameterValue(robot_description_content, value_type=str)
    }

    # ----- Nodes -----

    # Robot State Publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': use_sim_time}],
    )

    # Gazebo server only. Running GUI separately is more robust because a GUI
    # crash no longer tears down the simulation server.
    gazebo_server = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py'
        ]),
        launch_arguments={
            'gz_args': [
                '-s -r ',
                '--headless-rendering ',
                '-v ', gz_verbosity, ' ',
                world_file,
            ],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=IfCondition(headless_rendering),
    )
    gazebo_server_no_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            FindPackageShare('ros_gz_sim'), '/launch/gz_sim.launch.py'
        ]),
        launch_arguments={
            'gz_args': ['-s -r -v ', gz_verbosity, ' ', world_file],
            'on_exit_shutdown': 'true',
        }.items(),
        condition=UnlessCondition(headless_rendering),
    )

    # Optional GUI attaches to the running server. If it crashes, the server
    # keeps running and the rest of the stack is unaffected.
    gazebo_gui = TimerAction(
        period=2.0,
        actions=[
            ExecuteProcess(
                cmd=['gz', 'sim', '-g', '-v', gz_verbosity, '--force-version', '8'],
                output='screen',
            )
        ],
        condition=IfCondition(gui),
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
        arguments=['gripper_controller', '--controller-manager', '/controller_manager'],
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
            # Single overhead camera
            '/camera_overhead@sensor_msgs/msg/Image@gz.msgs.Image',
            '/camera_overhead/camera_info@sensor_msgs/msg/CameraInfo@gz.msgs.CameraInfo',
            '/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock',
        ],
        remappings=[
            ('/camera_overhead', '/camera_overhead/image_raw'),
        ],
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
    )

    # Perception launch (delayed to let cameras initialize)
    perception_launch = TimerAction(
        period=2.5,
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
        period=4.5,
        actions=[
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    os.path.join(control_share, 'launch', 'control.launch.py')
                ),
                launch_arguments={'use_sim_time': 'true'}.items(),
            )
        ],
    )

    # rqt_image_view for overhead camera feed
    rqt_overhead = TimerAction(
        period=4.0,
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'run', 'rqt_image_view', 'rqt_image_view',
                     '/camera_overhead/processed'],
                output='screen',
            )
        ],
        condition=IfCondition(rqt),
    )

    return LaunchDescription([
        declare_use_sim_time,
        declare_world_file,
        declare_gui,
        declare_rqt,
        declare_headless,
        declare_gz_verbosity,
        robot_state_publisher,
        gazebo_server,
        gazebo_server_no_headless,
        gazebo_gui,
        spawn_robot,
        spawn_joint_state_broadcaster,
        delayed_arm_controller,
        delayed_gripper_controller,
        ros_gz_bridge,
        perception_launch,
        control_launch,
        rqt_overhead,
    ])
