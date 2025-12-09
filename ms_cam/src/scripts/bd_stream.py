import cv2
import numpy as np
import os

# Load calibration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAL_PATH = os.path.join(SCRIPT_DIR, "..", "camera_calibration.npz")
cal = np.load(CAL_PATH)
print(cal.files)

camera_matrix = cal["K"]
dist_coeffs = cal["dist"]

# Open the camera
cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
cap.set(cv2.CAP_PROP_FPS, 30)

# Precompute undistortion maps
ret, frame = cap.read()
h, w = frame.shape[:2]
new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

# Set up blob detector parameters
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 300
params.maxArea = 50000
params.filterByCircularity = False
params.filterByInertia = False
params.filterByConvexity = False
params.filterByColor = False

detector = cv2.SimpleBlobDetector_create(params)

# HSV range for pink
lower_pink = np.array([140, 50, 50])
upper_pink = np.array([180, 255, 255])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break
    
    frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR) # undistort

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_pink, upper_pink)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Detect blobs in mask
    keypoints = detector.detect(mask)

    # If any blobs detected, find the largest
    if keypoints:
        # Sort by blob size (keypoint.size is diameter)
        largest = max(keypoints, key=lambda k: k.size)
        x, y = int(largest.pt[0]), int(largest.pt[1])

        pts = np.array([[[x, y]]], dtype=np.float32)
        undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=new_camera_matrix)
        xn, yn = undistorted[0][0]

        print(f"Centroid pixel: ({x}, {y})  →  normalized: ({xn:.3f}, {yn:.3f})")


        # Draw largest blob
        output = cv2.drawKeypoints(
            frame, [largest], None, (255, 0, 255),
            cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        )

        # Optionally mark the centroid with a small circle
        cv2.circle(output, (x, y), 5, (0, 255, 0), -1)
    else:
        output = frame

    cv2.imshow("Pink Blob Detection", output)
    cv2.imshow("Pink Mask", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
