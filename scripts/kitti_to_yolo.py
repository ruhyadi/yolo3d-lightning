"""
Convert KITTI format to YOLO format.
"""

import os
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

from typing import Tuple


class KITTI2YOLO:
    def __init__(
        self,
        dataset_path: str = "../data/KITTI",
        classes: Tuple = ["car", "van", "truck", "pedestrian", "cyclist"],
        img_width: int = 1224,
        img_height: int = 370,
    ):

        self.dataset_path = dataset_path
        self.img_width = img_width
        self.img_height = img_height
        self.classes = classes
        self.ids = {self.classes[i]: i for i in range(len(self.classes))}

        # create new directory
        self.label_path = os.path.join(self.dataset_path, "labels")
        if not os.path.isdir(self.label_path):
            os.makedirs(self.label_path)
        else:
            print("[INFO] Directory already exist...")

    def convert(self):
        files = glob(os.path.join(self.dataset_path, "label_2", "*.txt"))
        for file in tqdm(files):
            with open(file, "r") as f:
                filename = os.path.join(self.label_path, file.split("/")[-1])
                dump_txt = open(filename, "w")
                for line in f:
                    parse_line = self.parse_line(line)
                    if parse_line["name"].lower() not in self.classes:
                        continue

                    xmin, ymin, xmax, ymax = parse_line["bbox_camera"]
                    xcenter = ((xmax - xmin) / 2 + xmin) / self.img_width
                    if xcenter > 1.0:
                        xcenter = 1.0
                    ycenter = ((ymax - ymin) / 2 + ymin) / self.img_height
                    if ycenter > 1.0:
                        ycenter = 1.0
                    width = (xmax - xmin) / self.img_width
                    if width > 1.0:
                        width = 1.0
                    height = (ymax - ymin) / self.img_height
                    if height > 1.0:
                        height = 1.0

                    bbox_yolo = f"{self.ids[parse_line['name'].lower()]} {xcenter:.3f} {ycenter:.3f} {width:.3f} {height:.3f}"
                    dump_txt.write(bbox_yolo + "\n")

                dump_txt.close()

    def parse_line(self, line):
        parts = line.split(" ")
        output = {
            "name": parts[0].strip(),
            "xyz_camera": (float(parts[11]), float(parts[12]), float(parts[13])),
            "wlh": (float(parts[9]), float(parts[10]), float(parts[8])),
            "yaw_camera": float(parts[14]),
            "bbox_camera": (
                float(parts[4]),
                float(parts[5]),
                float(parts[6]),
                float(parts[7]),
            ),
            "truncation": float(parts[1]),
            "occlusion": float(parts[2]),
            "alpha": float(parts[3]),
        }

        # Add score if specified
        if len(parts) > 15:
            output["score"] = float(parts[15])
        else:
            output["score"] = np.nan

        return output


if __name__ == "__main__":

    # argparser
    parser = argparse.ArgumentParser(description="KITTI to YOLO Convertion")
    parser.add_argument("--dataset_path", type=str, default="../data/KITTI")
    parser.add_argument(
        "--classes",
        type=Tuple,
        default=["car", "van", "truck", "pedestrian", "cyclist"],
    )
    parser.add_argument("--img_width", type=int, default=1224)
    parser.add_argument("--img_height", type=int, default=370)
    args = parser.parse_args()

    kitit2yolo = KITTI2YOLO(
        dataset_path=args.dataset_path,
        classes=args.classes,
        img_width=args.img_width,
        img_height=args.img_height,
    )
    kitit2yolo.convert()
