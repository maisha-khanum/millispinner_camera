import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data
import cv2

class ImageSaver(Node):
    def __init__(self):
        super().__init__('image_saver')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/image',
            self.callback,
            qos_profile_sensor_data
        )
        self.counter = 0

    def callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(f"./calibration/images/calib_img_{self.counter:03d}.png", cv_image)
        self.get_logger().info(f"Saved frame {self.counter}")
        self.counter += 1
        if self.counter >= 30:
            rclpy.shutdown()

rclpy.init()
rclpy.spin(ImageSaver())
