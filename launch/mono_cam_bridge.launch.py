"""
mono_cam_bridge.launch.py

Bridges the mono_cam camera topics from Gazebo Harmonic into ROS2.

Usage (after starting PX4 SITL with x500_mono_cam):

  source /opt/ros/humble/setup.bash
  ros2 launch ~/PX4-Autopilot/launch/mono_cam_bridge.launch.py

Then in rqt: Plugins -> Visualization -> Image View -> /mono_cam/image_raw
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # The Gazebo world name is used as a namespace prefix for gz transport topics.
    # For PX4 SITL this is typically 'default' (i.e. topics are just /mono_cam/image).
    # If you use a custom world, override with:  ros2 launch ... world:=<your_world>
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='',
        description='Gazebo world name prefix (leave empty for default PX4 SITL worlds)'
    )

    world = LaunchConfiguration('world')

    # ros_gz_bridge: bridges raw image data
    image_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='mono_cam_image_bridge',
        arguments=[
            '/mono_cam/image@sensor_msgs/msg/Image[gz.msgs.Image',
        ],
        remappings=[
            ('/mono_cam/image', '/mono_cam/image_raw'),
        ],
        output='screen',
    )

    # ros_gz_bridge: bridges camera info
    camera_info_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='mono_cam_info_bridge',
        arguments=[
            '/mono_cam/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo',
        ],
        output='screen',
    )

    return LaunchDescription([
        world_arg,
        image_bridge,
        camera_info_bridge,
    ])
