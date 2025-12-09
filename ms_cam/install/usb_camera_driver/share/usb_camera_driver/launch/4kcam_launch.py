from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Dynamically find the path to the config file within the package
    config_path = os.path.join(
        get_package_share_directory('ros2_logitech_brio_publisher'),
        'config',
        '4kcam.yaml'
    )

    return LaunchDescription([
        Node(
            package='usb_camera_driver',
            executable='usb_camera_driver_node',
            name='usb_camera_driver',
            parameters=[
                {'camera_id': 2},
                {'image_width': 3840},
                {'image_height': 2160},
                {'framerate': 30.0},
                {'pixel_format': 'mjpeg'},
                {'camera_info_url': f'file://{config_path}'}
            ]
        )
    ])
