""" Inference Code """

from typing import List
from PIL import Image
import cv2
from glob import glob
import numpy as np

import torch
from torchvision.transforms import transforms
from pytorch_lightning import LightningModule
from src.utils import Calib
from src.utils.ClassAverages import ClassAverages
from src.utils.Plotting import calc_alpha, plot_3d_box
from src.utils.Math import calc_location
from src.utils.Plotting import calc_theta_ray
from src.utils.Plotting import Plot3DBoxBev

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils

import onnxruntime
import openvino.runtime as ov

from src.utils.math2 import compute_orientaion, detectionInfo, recover_angle, translation_constraints

dotenv.load_dotenv(override=True)
log = src.utils.get_pylogger(__name__)
root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference(config: DictConfig):

    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']

    # use global calib file
    proj_matrix = Calib.get_P(config.get("calib_file"))

    # Averages Dimension list
    class_averages = ClassAverages()

    # init detector model
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)

    # init regressor model
    if config.get("inference_type") == "pytorch":
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights")))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU') #TODO: change to config.get("device")
        infer_req = regressor.create_infer_request()

    # init preprocessing
    log.info(f"Instantiating preprocessing")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    if not os.path.exists(config.get("output_dir")):
        os.makedirs(config.get("output_dir"))

    # Initialize plotting module
    plot3dbev = Plot3DBoxBev(proj_matrix)

    # TODO: able inference on videos
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        name = img_path.split("/")[-1].split(".")[0]
        img = Image.open(img_path)
        img_draw = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2RGB)
        # detect object with Detector
        dets = detector(img).crop(save=config.get("save_det2d"))
        # TODO: remove DIMS
        DIMS = []
        # txt results
        RESULTS_TXT = []
        for det in dets:
            obj = detectionInfo(line = np.zeros(15))
            box = [box.cpu().numpy() for box in det["box"]]  # xyxy
            obj.xmin, obj.ymin, obj.xmax, obj.ymax = box[0], box[1], box[2], box[3]
            obj.name = det["label"].split(" ")[0]

            # preprocess img with torch.transforms
            crop = preprocess(cv2.resize(det["im"], (224, 224)))
            # batching img
            crop = crop.reshape((1, *crop.shape)).to(config.get("device"))
            # regress 2D bbox
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)
                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]
            # refinement dimension
            try:
                dim += class_averages.get_item(class_to_labels(det["cls"].cpu().numpy()))
                DIMS.append(dim)
            except:
                dim = DIMS[-1]

            # assign dimension
            obj.h, obj.w, obj.l = dim[0], dim[1], dim[2]
            obj.alpha = recover_angle(orient, conf, 2)
            obj.rot_global, rot_local = compute_orientaion(proj_matrix, obj)
            obj.tx, obj.ty, obj.tz = translation_constraints(proj_matrix, obj, rot_local)

            output_line = obj.member_to_list()
            output_line.append(1.0)
            output_line = " ".join([str(x) for x in output_line]) + "\n"
            
            # write results
            with open(f"{config.get('output_dir')}/{name}.txt", "a") as f:
                f.write(output_line)

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


if __name__ == "__main__":

    inference()
