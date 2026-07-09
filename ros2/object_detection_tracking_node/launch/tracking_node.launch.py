from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Default config path resolved through the package share.
    default_config = PathJoinSubstitution([
        FindPackageShare("object_detection_tracking_node"),
        "config",
        "default.yaml",
    ])

    return LaunchDescription([
        DeclareLaunchArgument(
            "image_topic",
            default_value="/camera/image_raw",
            description="ROS2 image topic to subscribe to.",
        ),
        DeclareLaunchArgument(
            "model_path",
            default_value="runs/segment/output/training/yolo_training/weights/best.pt",
            description="Path to the YOLO segmentation model weights (relative to workspace/project root or absolute).",
        ),
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Path to node parameter YAML file. Default resolves to the package share.",
        ),
        DeclareLaunchArgument(
            "rosbag_path",
            default_value="",
            description="Optional path to a ROS2 bag. If provided, the launch file will run 'ros2 bag play' before the node.",
        ),
        DeclareLaunchArgument(
            "play_rosbag",
            default_value="true",
            description="Whether to start ros2 bag play when rosbag_path is provided.",
        ),

        # Optionally play a ROS2 bag so the node has something to consume.
        ExecuteProcess(
            condition=IfCondition(
                PythonExpression([
                    "'", LaunchConfiguration("rosbag_path"), "' != '' and '",
                    LaunchConfiguration("play_rosbag"), "' == 'true'"
                ])
            ),
            cmd=["ros2", "bag", "play", LaunchConfiguration("rosbag_path")],
            output="screen",
        ),

        Node(
            package="object_detection_tracking_node",
            executable="tracking_node",
            name="object_detection_tracking_node",
            output="screen",
            parameters=[
                LaunchConfiguration("config"),
                {
                    "image_topic": LaunchConfiguration("image_topic"),
                    "model_path": LaunchConfiguration("model_path"),
                },
            ],
        ),
    ])
