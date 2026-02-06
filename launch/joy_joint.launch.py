"""Launch file for joy_joint node (joint-space joystick control)."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    """Generate launch description for joy_joint node."""

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

    joy_joint_node = Node(
        package='joy_arm',
        executable='joy_joint_node',
        name='joy_joint_node',
        output='screen',
        parameters=[{
            'deadzone': LaunchConfiguration('deadzone'),
            'control_rate': LaunchConfiguration('control_rate'),
            'gripper_close_button': LaunchConfiguration('gripper_close_button'),
            'gripper_open_button': LaunchConfiguration('gripper_open_button'),
        }]
    )

    return LaunchDescription([
        deadzone_arg,
        control_rate_arg,
        gripper_close_button_arg,
        gripper_open_button_arg,
        joy_joint_node,
    ])
