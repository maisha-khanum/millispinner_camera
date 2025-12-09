import cv2
import numpy as np
import os
import time
import pickle
import lz4.frame
import argparse
import joblib


class KalmanBlobTracker:
    def __init__(self, dt=1/30):
        self.dt = dt

        # State: [x, y, vx, vy]
        self.x = np.zeros((4,1))

        # State covariance
        self.P = np.eye(4) * 1.0

        # Dynamics matrix
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ])

        # Process noise
        # q = 0.01
        # self.Q = np.eye(4) * q
        self.Q = np.diag([1e-4, 1e-4, 5e-3, 5e-3])

        # Measurement model
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])

        # Measurement noise
        r = 0.002
        self.R = np.eye(2) * r

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x

    def update(self, z):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        y = z - self.H @ self.x
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def skip_update(self):
        # Just ignore update step when occluded
        pass

def main():
    parser = argparse.ArgumentParser(description="Record 4K RGB frames with blob centroids to LZ4 file.")
    parser.add_argument("basename", type=str, nargs="?", help="(ignored) Base name of the output LZ4 file")
    args = parser.parse_args()

    # Output path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(SCRIPT_DIR, "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # Always save to fixed name
    output_file = os.path.join(recordings_dir, "_db.lz4")
    data_dict = []

    # Load camera calibration
    CAL_PATH = os.path.join(SCRIPT_DIR, "..", "camera_calibration.npz")
    cal = np.load(CAL_PATH)
    print("Calibration keys:", cal.files)

    camera_matrix = cal["K"]
    dist_coeffs = cal["dist"]

    # Open camera
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Precompute undistortion maps
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w,h), 1)
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv2.CV_32FC1)

    # Blob detector parameters
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 300
    params.maxArea = 50000
    detector = cv2.SimpleBlobDetector_create(params)

    lower_pink = np.array([140, 50, 50])
    upper_pink = np.array([180, 255, 255])

    # Kalman Filter tracker
    tracker = KalmanBlobTracker(dt=1/30)

    print(f"Recording to {output_file} ... Press 'q' to stop.")

    with lz4.frame.open(output_file, mode='wb') as f:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame not received. Exiting...")
                break

            # Undistort
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

            # Detect pink blob
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, lower_pink, upper_pink)
            mask = cv2.erode(mask, None, iterations=2)
            mask = cv2.dilate(mask, None, iterations=2)

            keypoints = detector.detect(mask)
            centroid_pixel = None
            centroid_norm = None

            tracker.predict()

            if keypoints: # grabs the largest
                largest = max(keypoints, key=lambda k: k.size)
                x, y = int(largest.pt[0]), int(largest.pt[1])
                centroid_pixel = (x, y)

                pts = np.array([[[x, y]]], dtype=np.float32)
                undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=new_camera_matrix)
                xn, yn = undistorted[0][0]
                centroid_norm = (float(xn), float(yn))

                frame = cv2.drawKeypoints(frame, [largest], None, (255, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

                print(f"Centroid pixel: ({x}, {y})  →  normalized: ({xn:.3f}, {yn:.3f})")

            # if centroid_norm is not None:
            #     tracker.update(np.array([[xn], [yn]]))
            # else:
            #     tracker.skip_update()

            pred_x = int(tracker.x[0])
            pred_y = int(tracker.x[1])
            cv2.circle(frame, (pred_x, pred_y), 6, (0, 255, 255), 2)


            pred = tracker.H @ tracker.x
            y = np.array([[xn], [yn]]) - pred
            if np.linalg.norm(y) < 0.1:    # tune this gate
                tracker.update(np.array([[xn],[yn]]))
            else:
                tracker.skip_update()

            # Timestamp
            timestamp_ns = int(time.time() * 1e9)

            # Serialize frame + centroid
            data_dict.append({
                "timestamp_ns": timestamp_ns,
                "frame": frame,
                "mask": mask,
                "centroid_pixel": centroid_pixel,
                "centroid_norm": centroid_norm
            })


            # Preview
            cv2.imshow("Pink Blob Detection", cv2.resize(frame, (1280, 720)))
            cv2.imshow("Pink Mask", cv2.resize(mask, (640, 360)))

            # print(f"Centroid pixel: ({centroid_pixel})  →  normalized: ({centroid_norm})")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Recording stopped.")
                break
    
    # Save all frames to disk using joblib + LZ4
    print("Saving to disk...")
    t_start = time.time()
    joblib.dump(data_dict, output_file, compress=('lz4', 1))
    print(f"[INFO] Done saving, took {time.time() - t_start:.2f}s")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
