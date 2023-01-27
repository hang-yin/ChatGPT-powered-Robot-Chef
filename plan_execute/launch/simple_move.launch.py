from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():

    test_node = Node(
        package='plan_execute',
        executable='simple_move',
        output='screen'
    )

    launch_franka = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
                PathJoinSubstitution([
                    FindPackageShare('franka_moveit_config'),
                    'launch/moveit.launch.py'
                ])
            ]),
        launch_arguments=[('robot_ip', 'dont-care'), ('use_fake_hardware', 'true')]
    )

    return LaunchDescription([
        launch_franka,
        test_node
    ])
