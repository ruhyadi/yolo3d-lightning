""" Evaluating Code """

from typing import List
import cv2
from glob import glob
import numpy as np

import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from src.utils import Calib
from src.utils.averages import ClassAverages
from src.utils.Math import (
    compute_orientaion,
    recover_angle,
    translation_constraints,
)

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject
from tqdm import tqdm

import src.utils.kitti_common as kitti
from src.utils.eval import get_official_eval_result

log = src.utils.get_pylogger(__name__)

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="evaluate.yaml")
def evaluate(config: DictConfig):
    """Inference function"""
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
    class_averages = ClassAverages()

    # initialize detector model
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)

    log.info(f"Instantiating regressor <{config.model._target_}>")
    regressor: LightningModule = hydra.utils.instantiate(config.model)
    regressor.load_state_dict(torch.load(config.get("regressor_weights")))
    regressor.eval().to(config.get("device"))

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("pred_dir"), exist_ok=True)

    # evalaution on validation sets
    imgs_path = images_sets(config.get("val_images_path"), config.get("val_sets"))
    for img_path in tqdm(imgs_path):
        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detect object with Detector
        dets = detector(img).crop(save=config.get("save_det2d"))
        # dimension averages #TODO: depricated
        DIMENSION = []
        # loop thru detections
        for det in dets:
            # initialize object container
            obj = KITTIObject()
            obj.name = det["label"].split(" ")[0].capitalize()
            obj.truncation = float(0.00)
            obj.occlusion = int(-1)
            box = [box.cpu().numpy() for box in det["box"]]
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = box[0], box[1], box[2], box[3]

            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))

            # regress 2D bbox with Regressor
            [orient, conf, dim] = regressor(crop)
            orient = orient.cpu().detach().numpy()[0, :, :]
            conf = conf.cpu().detach().numpy()[0, :]
            dim = dim.cpu().detach().numpy()[0, :]

            # dimension averages # TODO: depricated
            try:
                dim += class_averages.get_item(class_to_labels(det["cls"].cpu().numpy()))
                DIMENSION.append(dim)
            except:
                dim = DIMENSION[-1]

            obj.alpha = recover_angle(orient, conf, 2)
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.rot_global, rot_local = compute_orientaion(P2, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(P2, obj, rot_local)
            obj.score = np.round(det["conf"].cpu().numpy(), 2)

            # output prediction label
            output_line = obj.member_to_list()
            output_line = " ".join([str(i) for i in output_line]) + "\n"

            # write results
            with open(f"{config.get('pred_dir')}/{img_name}.txt", "a") as f:
                f.write(output_line)

    # evaluate results
    log.info(f"Evaluating results")
    results = get_evaluation(config.get("pred_dir"), config.get("gt_dir"))
    log.info(f"Results: {results}")

    return results


def detector_yolov5(model_path: str, cfg_path: str, classes: int, device: str):
    """YOLOv5 detector model"""
    sys.path.append(str(root / "yolov5"))

    # NOTE: ignore import error
    from models.common import AutoShape
    from models.yolo import Model
    from utils.general import intersect_dicts
    from utils.torch_utils import select_device

    device = select_device(
        ("0" if torch.cuda.is_available() else "cpu") if device is None else device
    )

    model = Model(cfg_path, ch=3, nc=classes)
    ckpt = torch.load(model_path, map_location=device)  # load
    csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
    csd = intersect_dicts(csd, model.state_dict(), exclude=["anchors"])  # intersect
    model.load_state_dict(csd, strict=False)  # load
    if len(ckpt["model"].names) == classes:
        model.names = ckpt["model"].names  # set class names attribute
    model = AutoShape(model)  # for file/URI/PIL/cv2/np inputs and NMS

    return model.to(device)


def class_to_labels(class_: int, list_labels: List = None):

    if list_labels is None:
        # TODO: change some labels mistakes
        list_labels = ["car", "car", "truck", "pedestrian", "cyclist"]

    return list_labels[int(class_)]


def get_evaluation(pred_path: str, gt_path: str):
    """Evaluate results"""
    val_ids = [
        int(res.split("/")[-1].split(".")[0])
        for res in sorted(glob(os.path.join(pred_path, "*.txt")))
    ]
    pred_annos = kitti.get_label_annos(pred_path, val_ids)
    gt_annos = kitti.get_label_annos(gt_path, val_ids)

    # compute mAP
    results = get_official_eval_result(
        gt_annos=gt_annos, dt_annos=pred_annos, current_classes=0
    )

    return results


def read_sets(path: str):
    """Read validation sets"""
    with open(path, "r") as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def generate_sets(path: str):
    """Generate validation sets from images path"""
    pred_results = [res.split("/")[-1].split(".")[0] for res in glob(path, "*.txt")]


def images_sets(imgs_path: str, sets_path: str):
    """Read images sets"""
    imgs_path = sorted(glob(os.path.join(imgs_path, "*")))
    val_sets = read_sets(sets_path)
    return [
        img_path
        for img_path in imgs_path
        if int(img_path.split("/")[-1].split(".")[0]) in val_sets
    ]


if __name__ == "__main__":
    
    evaluate()
