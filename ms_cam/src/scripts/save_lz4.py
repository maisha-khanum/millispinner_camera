import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions
from rosbag2_py._storage import TopicMetadata
from rclpy.serialization import serialize_message
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.qos import qos_profile_sensor_data

import os

class ImageSaver(Node):
    def __init__(self, bag_path='image_bag_lz4'):
        super().__init__('image_saver')

        # Configure bag writer
        self.writer = SequentialWriter()
        storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
        converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
        self.writer.open(storage_options, converter_options)

        # Register the /image topic with correct metadata
        topic_metadata = TopicMetadata(
            name='/image',
            type='sensor_msgs/msg/Image',
            serialization_format='cdr',
            offered_qos_profiles=''  # leave empty for default
        )
        self.writer.create_topic(topic_metadata)

        # qos_profile = QoSProfile(
        #     reliability=ReliabilityPolicy.RELIABLE,
        #     depth=10
        # )

        # self.subscription = self.create_subscription(
        #     Image,
        #     '/image',
        #     self.callback,
        #     qos_profile
        # )

        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.callback,
            qos_profile_sensor_data
        )

    def callback(self, msg):
        timestamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.get_logger().info(f"Saving image at {timestamp_ns} ns")
        self.writer.write('/image', serialize_message(msg), timestamp_ns)

def main():
    rclpy.init()
    bag_name = './src/data/image_bag_lz41'
    if os.path.exists(bag_name):
        print(f"Bag '{bag_name}' already exists. Delete or rename it first.")
        return
    node = ImageSaver(bag_path=bag_name)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
