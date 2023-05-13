"""
Generate frames to vid
Usage:
python scripts/frames_to_video.py \
    --imgs_path /path/to/imgs \
    --vid_path /path/to/vid \
    --fps 24 \
    --frame_size 1242 375 \
    --resize

python scripts/frames_to_video.py \
    --imgs_path outputs/2023-05-13/22-51-34/inference \
    --vid_path tmp/output_videos/001.mp4 \
    --fps 3 \
    --frame_size 1550 387 \
    --resize
"""

import argparse
import cv2
from glob import glob
import os
from tqdm import tqdm

def generate(imgs_path, vid_path, fps=30, frame_size=(1242, 375), resize=True):
    """Generate frames to vid"""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_writer = cv2.VideoWriter(vid_path, fourcc, fps, frame_size)
    imgs_glob = sorted(glob(os.path.join(imgs_path, "*.png")))
    if resize:
        for img_path in tqdm(imgs_glob):
            img = cv2.imread(img_path)
            img = cv2.resize(img, frame_size)
            vid_writer.write(img)
    else:
        for img_path in imgs_glob:
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            vid_writer.write(img)
    vid_writer.release()
    print('[INFO] Video saved to {}'.format(vid_path))

if __name__ == "__main__":
    # create argparser
    parser = argparse.ArgumentParser(description="Generate frames to vid")
    parser.add_argument("--imgs_path", type=str, default="outputs/2022-10-23/21-03-50/inference", help="path to imgs")
    parser.add_argument("--vid_path", type=str, default="outputs/videos/004.mp4", help="path to vid")
    parser.add_argument("--fps", type=int, default=24, help="fps")
    parser.add_argument("--frame_size", type=int, nargs=2, default=(int(1242), int(375)), help="frame size")
    parser.add_argument("--resize", action="store_true", help="resize")
    args = parser.parse_args()

    # generate vid
    generate(args.imgs_path, args.vid_path, args.fps, args.frame_size)