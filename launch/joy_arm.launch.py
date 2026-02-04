"""Launch file for joy_arm node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for joy_arm node."""

    # Declare launch arguments
    linear_scale_arg = DeclareLaunchArgument(
        'linear_scale',
        default_value='0.05',
        description='Linear velocity scale (m/s per joystick unit)'
    )

    angular_scale_arg = DeclareLaunchArgument(
        'angular_scale',
        default_value='0.2',
        description='Angular velocity scale (rad/s per joystick unit)'
    )

    deadzone_arg = DeclareLaunchArgument(
        'deadzone',
        default_value='0.1',
        description='Joystick deadzone threshold'
    )

    control_rate_arg = DeclareLaunchArgument(
        'control_rate',
        default_value='10.0',
        description='Control loop rate (Hz)'
    )

    gripper_close_button_arg = DeclareLaunchArgument(
        'gripper_close_button',
        default_value='0',
        description='Button index to close gripper'
    )

    gripper_open_button_arg = DeclareLaunchArgument(
        'gripper_open_button',
        default_value='1',
        description='Button index to open gripper'
    )

    max_velocity_scaling_arg = DeclareLaunchArgument(
        'max_velocity_scaling',
        default_value='0.5',
        description='Max velocity scaling factor for motion planning'
    )

    max_acceleration_scaling_arg = DeclareLaunchArgument(
        'max_acceleration_scaling',
        default_value='0.3',
        description='Max acceleration scaling factor for motion planning'
    )

    # Node
    joy_arm_node = Node(
        package='joy_arm',
        executable='joy_arm_node',
        name='joy_arm_node',
        output='screen',
        parameters=[{
            'linear_scale': LaunchConfiguration('linear_scale'),
            'angular_scale': LaunchConfiguration('angular_scale'),
            'deadzone': LaunchConfiguration('deadzone'),
            'control_rate': LaunchConfiguration('control_rate'),
            'gripper_close_button': LaunchConfiguration('gripper_close_button'),
            'gripper_open_button': LaunchConfiguration('gripper_open_button'),
            'max_velocity_scaling': LaunchConfiguration('max_velocity_scaling'),
            'max_acceleration_scaling': LaunchConfiguration('max_acceleration_scaling'),
        }]
    )

    return LaunchDescription([
        linear_scale_arg,
        angular_scale_arg,
        deadzone_arg,
        control_rate_arg,
        gripper_close_button_arg,
        gripper_open_button_arg,
        max_velocity_scaling_arg,
        max_acceleration_scaling_arg,
        joy_arm_node,
    ])
