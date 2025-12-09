import cv2
import numpy as np
import rclpy
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rosbag2_py import StorageFilter

# === Configuration ===
bag_path = '../ms_cam/src/data/image_bag_lz41'
video_out_path = 'pink_blob_output.mp4'
image_topic = '/image'
bridge = CvBridge()

# === Setup blob detector ===
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 300
params.maxArea = 5000
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False
detector = cv2.SimpleBlobDetector_create(params)

# === HSV pink color range ===
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([180, 255, 255])

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([170, 100, 100])
upper_red2 = np.array([180, 255, 255])


# === Initialize ROS and reader ===
rclpy.init()
reader = SequentialReader()
storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
reader.open(storage_options, converter_options)

# === Get topic types ===
topic_types = reader.get_all_topics_and_types()
type_map = {t.name: t.type for t in topic_types}
filter = StorageFilter(topics=['/image'])
reader.set_filter(filter)

# === Read first frame to get size ===
first_frame = None
fps = 30  # You can adjust if needed
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic == image_topic:
        msg = deserialize_message(data, Image)
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        height, width = frame.shape[:2]
        first_frame = frame
        break

if first_frame is None:
    print("No image messages found.")
    exit()

# === Set up video writer ===
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_color = cv2.VideoWriter('pink_blob_output.mp4', fourcc, fps, (width, height))
out_mask = cv2.VideoWriter('pink_blob_mask.mp4', fourcc, fps, (width, height), isColor=False)

# === Replay bag and process ===
reader.seek(0)
while reader.has_next():
    topic, data, timestamp = reader.read_next()
    if topic != image_topic:
        continue

    msg = deserialize_message(data, Image)
    frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    keypoints = detector.detect(mask)

    if keypoints:
        largest = max(keypoints, key=lambda k: k.size)
        x, y = int(largest.pt[0]), int(largest.pt[1])

        # Create mask of largest blob only
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        largest_mask = np.zeros_like(mask)

        for label in range(1, num_labels):
            cx, cy = centroids[label]
            if np.linalg.norm([cx - x, cy - y]) < 5:  # centroid close to blob center
                largest_mask[labels == label] = 255
                break

        # Annotated frame
        output = cv2.drawKeypoints(frame, [largest], None, (255, 0, 255),
                                   cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv2.circle(output, (x, y), 5, (0, 0, 255), -1)
    else:
        output = frame
        largest_mask = np.zeros_like(mask)

    out_color.write(output)
    out_mask.write(largest_mask)

out_color.release()
out_mask.release()
print("Saved videos: pink_blob_output.mp4 and pink_blob_mask.mp4")