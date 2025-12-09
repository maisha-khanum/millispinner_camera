import cv2
import joblib
import time
import os

def main():
    cam_index = 1  # /dev/video1
    width, height = 3840, 2160
    fps = 30

    # Open camera
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: Could not access camera.")
        return

    # Configure camera
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"🎥 Recording {actual_width}x{actual_height} @ {actual_fps} fps")

    # Prepare output path
    recordings_dir = os.path.join(os.path.dirname(__file__), "..", "recordings")
    os.makedirs(recordings_dir, exist_ok=True)
    output_file = os.path.join(recordings_dir, "_vid.lz4")  # Joblib file

    data_dict = []
    print(f"Recording to {output_file} ... Press 'q' to stop.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received. Exiting...")
            break

        # Timestamp (nanoseconds)
        timestamp_ns = int(time.time() * 1e9)

        # Store frame and metadata
        data_dict.append({
            "timestamp_ns": timestamp_ns,
            "frame": frame
        })

        # Downscaled preview
        cv2.imshow("4K Preview", cv2.resize(frame, (1280, 720)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("✅ Recording stopped.")
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
