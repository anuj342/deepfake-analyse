import cv2
import os
from mtcnn import MTCNN
import glob

# --- CONFIGURATION ---
# Path to the folder containing the original Celeb-DF videos
SOURCE_VIDEOS_PATH = 'Celeb-DF'

# Path to the folder where we will save the processed face images
OUTPUT_FACES_PATH = 'processed_data/train'

# Desired size for the final face images
IMAGE_SIZE = 224  # (Or 299 for XceptionNet)

# How many frames to skip between captures. E.g., 10 means process 1 frame every 10.
FRAME_INTERVAL = 10

# --- END CONFIGURATION ---

def preprocess_videos():
    """
    Main function to loop through videos, extract faces, and save them.
    """
    # Initialize the MTCNN face detector
    print("Initializing MTCNN face detector...")
    detector = MTCNN()
    print("Initialization complete.")

    # Create the output directories if they don't exist
    os.makedirs(os.path.join(OUTPUT_FACES_PATH, 'real'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_FACES_PATH, 'fake'), exist_ok=True)

    # Define the video folders and their corresponding labels
    video_sources = {
        os.path.join(SOURCE_VIDEOS_PATH, 'Celeb-real'): 'real',
        os.path.join(SOURCE_VIDEOS_PATH, 'Celeb-synthesis'): 'fake',
        # Add other source folders if needed
    }

    for source_path, label in video_sources.items():
        print(f"\nProcessing videos from: {source_path} (Label: {label})")

        # Get a list of all video files in the source directory
        video_files = glob.glob(os.path.join(source_path, '*.mp4'))
        print(f"Found {len(video_files)} videos.")

        for video_path in video_files:
            process_single_video(video_path, label, detector)

def process_single_video(video_path, label, detector):
    """
    Extracts, detects, and saves faces from a single video file.
    """
    video_name = os.path.basename(video_path).split('.')[0]
    print(f"  -> Processing video: {video_name}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    Error: Could not open video {video_name}")
        return

    frame_count = 0
    saved_face_count = 0
    MAX_FRAMES = 30

    while True:
        if saved_face_count >= MAX_FRAMES:  # <--- stop after 30 faces
            break

        ret, frame = cap.read()
        if not ret:
            break  # End of video

        # Process frames at the specified interval
        if frame_count % FRAME_INTERVAL == 0:
            # MTCNN expects images in RGB format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces in the frame
            results = detector.detect_faces(frame_rgb)

            if results:
                # Get the bounding box of the most confident detection
                bounding_box = results[0]['box']
                x, y, w, h = bounding_box

                # Crop the face from the original BGR frame
                # Add a small margin to ensure the whole face is included
                margin = 20
                face = frame[max(0, y - margin):y + h + margin, max(0, x - margin):x + w + margin]

                if face.size > 0:
                    # Resize the face to the desired standard size
                    resized_face = cv2.resize(face, (IMAGE_SIZE, IMAGE_SIZE))

                    # Construct the output filename and save the image
                    output_filename = f"{video_name}_frame{frame_count}.jpg"
                    save_path = os.path.join(OUTPUT_FACES_PATH, label, output_filename)
                    cv2.imwrite(save_path, resized_face)
                    saved_face_count += 1

        frame_count += 1

    print(f"    -> Finished. Saved {saved_face_count} faces.")
    cap.release()

# --- SCRIPT EXECUTION ---
if __name__ == '__main__':
    preprocess_videos()