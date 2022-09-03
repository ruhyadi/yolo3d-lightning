"""Evaluate KITTI 3D Object Detection"""

import kitti_common as kitti
from eval import get_coco_eval_result, get_official_eval_result
import argparse

def evaluate(dt_path, gt_path, dt_sets):
    """Evaluate KITTI 3D Object Detection"""
    dt_annos = kitti.get_label_annos(dt_path)
    val_image_ids = _read_imageset_file(dt_sets)
    gt_annos = kitti.get_label_annos(gt_path, val_image_ids)

    print(get_official_eval_result(gt_annos, dt_annos, 0))
    # print(get_coco_eval_result(gt_annos, dt_annos, 0))

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description="Evaluate KITTI 3D Object Detection")
    parser.add_argument("--dt_path", type=str, default="/raid/didir/Repository/yolo3d-lightning/outputs/2022-09-03/17-30-28/inference", help="Path to detection results")
    parser.add_argument("--gt_path", type=str, default="/raid/didir/Repository/yolo3d-lightning/data/KITTI/label_2", help="Path to ground truth")
    parser.add_argument("--dt_sets", type=str, default="/raid/didir/Repository/yolo3d-lightning/data/KITTI/val_dummy.txt", help="Path to detection sets")
    args = parser.parse_args()

    # evaluate
    evaluate(args.dt_path, args.gt_path, args.dt_sets)