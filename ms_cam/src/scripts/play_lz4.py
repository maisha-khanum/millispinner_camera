import cv2
import joblib
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Manually browse 4K RGB frames from Joblib+LZ4 file using arrow keys.")
    parser.add_argument("basename", type=str, help="Base name of the LZ4 file to play (e.g., test_1)")
    args = parser.parse_args()

    # Build file path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    recordings_dir = os.path.join(SCRIPT_DIR, "..", "recordings")
    filename = args.basename
    if not filename.endswith(".lz4"):
        filename += ".lz4"
    input_file = os.path.join(recordings_dir, filename)

    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    print(f"Loading {input_file} using joblib...")
    try:
        data = joblib.load(input_file)
    except Exception as e:
        print(f"❌ Failed to load {input_file}: {e}")
        return

    print(f"✅ Loaded {len(data)} frames.")
    print("Use ↑ / ↓ to navigate frames, or 'q' to quit.")

    idx = 0
    total = len(data)

    while True:
        entry = data[idx]
        frame = entry.get("frame", None)
        mask = entry.get("mask", None)
        ts = entry.get("timestamp_ns", None)
        centroid_pixel = entry.get("centroid_pixel", None)
        centroid_norm = entry.get("centroid_norm", None)

        if frame is None:
            print(f"⚠️ Frame {idx} is empty.")
        else:
            display = cv2.resize(frame, (1280, 720))
            cv2.imshow("Playback", display)
        
        if frame is not None:
            display = cv2.resize(frame, (640, 360))
            cv2.imshow("Playback", display)

        # Print metadata in console
        print(f"\nFrame {idx+1}/{total}")
        if ts is not None:
            print(f"Timestamp: {ts}")
        if centroid_pixel or centroid_norm:
            print(f"Centroid pixel: {centroid_pixel}, normalized: {centroid_norm}")

        key = cv2.waitKey(0) & 0xFF  # Wait for key press

        if key == ord('q'):
            print("Playback stopped.")
            break
        elif key == 82:  # Up arrow
            idx = (idx + 1) % total
        elif key == 84:  # Down arrow
            idx = (idx - 1) % total
        else:
            print("Use ↑ to go forward, ↓ to go backward, or 'q' to quit.")

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
