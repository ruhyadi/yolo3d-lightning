"""
Convert video to frame
Usage:
python video_to_frame.py \
    --video_path /path/to/video \
    --output_path /path/to/output/folder \
    --fps 24

python scripts/video_to_frame.py \
    --video_path tmp/video/20230513_100429.mp4 \
    --output_path tmp/video_001 \
    --fps 20
"""

import argparse

import os
import cv2


def video_to_frame(video_path: str, output_path: str, fps: int = 5):
    """
    Convert video to frame

    Args:
        video_path: path to video
        output_path: path to output folder
        fps: how many frames per second to save 
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % fps == 0:
            cv2.imwrite(os.path.join(output_path, f"{frame_count:06d}.jpg"), frame)
        frame_count += 1

    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    video_to_frame(args.video_path, args.output_path, args.fps)