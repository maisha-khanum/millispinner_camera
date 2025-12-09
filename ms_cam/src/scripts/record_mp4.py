import cv2
import os
import argparse

def main():
    parser = argparse.ArgumentParser(description="Record 4K video from USB camera.")
    parser.add_argument("basename", type=str, help="Base name of the output MP4 file (e.g., test_1)")
    args = parser.parse_args()

    # Append .mp4 automatically
    filename = args.basename
    if not filename.endswith(".mp4"):
        filename += ".mp4"

    cam_index = 1  # /dev/video1
    width, height = 3840, 2160
    fps = 30

    # Open camera
    cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        print("ERROR: Could not access camera.")
        return

    # Set 4K resolution and FPS
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Ensure MJPG for high res
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    # Confirm actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(f"Recording {actual_width}x{actual_height} @ {actual_fps} fps")

    # Ensure recordings folder exists
    recordings_dir = os.path.join(os.path.dirname(__file__), "..", "recordings")
    os.makedirs(recordings_dir, exist_ok=True)

    # Full output path
    output_path = os.path.join(recordings_dir, filename)

    # Video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (actual_width, actual_height))

    print("Press 'q' to stop recording.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame not received. Exiting...")
            break

        out.write(frame)

        # Downscaled preview to speed up display
        cv2.imshow("4K Preview", cv2.resize(frame, (1280, 720)))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
