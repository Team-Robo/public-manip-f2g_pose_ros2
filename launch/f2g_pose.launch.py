from pathlib import Path
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare('f2g_pose_ros2')
    default_config = PathJoinSubstitution([pkg_share, 'config', 'f2g_pose.yaml'])

    return LaunchDescription([
        DeclareLaunchArgument(
            'auto_trigger', default_value='false', choices=['true', 'false'],
            description='Automatically request head segmentation at startup'
        ),
        DeclareLaunchArgument(
            'f2g_mode',
            default_value='fusion',
            description='F2G-Pose operation mode: "fusion" (Option A default) or "direct_head" (Option C)'
        ),
        DeclareLaunchArgument(
            'target_label',
            default_value='',
            description='Target object label for segmentation and grasp synthesis'
        ),
        DeclareLaunchArgument(
            'params_file',
            default_value=default_config,
            description='Path to YAML parameter configuration file'
        ),
        DeclareLaunchArgument(
            'device',
            default_value='cuda:0',
            description='Inference compute device'
        ),
        DeclareLaunchArgument(
            'head_color_topic',
            default_value='/camera/head_d455/color/image_raw',
            description='Head camera RGB image topic'
        ),
        DeclareLaunchArgument(
            'head_depth_topic',
            default_value='/camera/head_d455/depth/image_rect_raw',
            description='Head camera depth image topic'
        ),
        DeclareLaunchArgument(
            'head_mask_topic',
            default_value='/perc/head/mask',
            description='Head camera segmentation mask topic'
        ),
        DeclareLaunchArgument(
            'wrist_cloud_topic',
            default_value='/perc/grasp/cleaned_points',
            description='Wrist camera cleaned target point cloud topic'
        ),

        Node(
            package='f2g_pose_ros2',
            executable='f2g_pose_node',
            name='f2g_pose_node',
            output='screen',
            parameters=[
                LaunchConfiguration('params_file'),
                {
                    'f2g_mode': LaunchConfiguration('f2g_mode'),
                    'auto_trigger': ParameterValue(LaunchConfiguration('auto_trigger'), value_type=bool),
                    'target_label': LaunchConfiguration('target_label'),
                    'device': LaunchConfiguration('device'),
                    'head_color_topic': LaunchConfiguration('head_color_topic'),
                    'head_depth_topic': LaunchConfiguration('head_depth_topic'),
                    'head_mask_topic': LaunchConfiguration('head_mask_topic'),
                    'wrist_cloud_topic': LaunchConfiguration('wrist_cloud_topic'),
                }
            ]
        )
    ])
