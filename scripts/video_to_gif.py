"""Convert video to gif with moviepy"""

import argparse
import moviepy.editor as mpy

def generate(video_path, gif_path, fps):
    """Generate gif from video"""
    clip = mpy.VideoFileClip(video_path)
    clip.write_gif(gif_path, fps=fps)
    clip.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to gif")
    parser.add_argument("--video_path", type=str, default="outputs/videos/004.mp4", help="Path to video")
    parser.add_argument("--gif_path", type=str, default="outputs/gif/002.gif", help="Path to gif")
    parser.add_argument("--fps", type=int, default=5, help="GIF fps")
    args = parser.parse_args()

    # generate gif
    generate(args.video_path, args.gif_path, args.fps)