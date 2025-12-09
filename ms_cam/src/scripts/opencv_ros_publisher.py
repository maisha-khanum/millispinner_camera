from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node

class OpenCVCameraPublisher(Node):
    def __init__(self):
        super().__init__('opencv_camera_publisher')
        self.publisher = self.create_publisher(Image, '/image_raw', 10)
        self.bridge = CvBridge()
        self.timer = self.create_timer(1/30, self.publish_frame)

        self.cap = cv2.VideoCapture(2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

        # Add this for visual feedback
        cv2.imshow("OpenCV Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()


    def destroy_node(self):
        self.cap.release()
        super().destroy_node()
