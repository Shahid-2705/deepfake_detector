import os
import cv2
from tqdm import tqdm

# ================= CONFIG =================
RAW_VIDEO_DIR = "data/raw_videos"
OUTPUT_DIR = "data/extracted_frames"

REAL_FOLDERS = ["Celeb-real"]
FAKE_FOLDERS = ["Celeb-synthesis"]

FRAMES_PER_VIDEO = 8
# ==========================================


def extract_frames_from_video(video_path, output_folder, label_prefix):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return

    # Evenly spaced frame indices
    frame_indices = [
        int(total_frames * i / FRAMES_PER_VIDEO)
        for i in range(FRAMES_PER_VIDEO)
    ]

    video_name = os.path.splitext(os.path.basename(video_path))[0]

    saved_count = 0

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        if not ret:
            continue

        filename = f"{label_prefix}_{video_name}_{saved_count}.jpg"
        save_path = os.path.join(output_folder, filename)

        cv2.imwrite(save_path, frame)
        saved_count += 1

    cap.release()


def process_folder(folder_name, label_type):
    input_dir = os.path.join(RAW_VIDEO_DIR, folder_name)
    output_dir = os.path.join(OUTPUT_DIR, label_type)

    video_files = [
        f for f in os.listdir(input_dir)
        if f.endswith(".mp4")
    ]

    for video_file in tqdm(video_files, desc=f"Processing {folder_name}"):
        video_path = os.path.join(input_dir, video_file)
        extract_frames_from_video(video_path, output_dir, label_type)


def main():
    print("\nStarting full frame extraction...\n")

    # Real videos
    for folder in REAL_FOLDERS:
        process_folder(folder, "real")

    # Fake videos
    for folder in FAKE_FOLDERS:
        process_folder(folder, "fake")

    print("\nFrame extraction completed.\n")


if __name__ == "__main__":
    main()
