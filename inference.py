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
from src.utils.averages import ClassAverages
from src.utils.Plotting import calc_alpha, plot_3d_box
from src.utils.Math import calc_location, compute_orientaion, recover_angle, translation_constraints
from src.utils.Plotting import calc_theta_ray
from src.utils.Plotting import Plot3DBoxBev

import dotenv
import hydra
from omegaconf import DictConfig
import os
import sys
import pyrootutils
import src.utils
from src.utils.utils import KITTIObject

import time

log = src.utils.get_pylogger(__name__)

try: 
    import onnxruntime
    import openvino.runtime as ov
except ImportError:
    log.warning("ONNX and OpenVINO not installed")

dotenv.load_dotenv(override=True)

root = pyrootutils.setup_root(__file__, dotenv=True, pythonpath=True)

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference(config: DictConfig):
    """Inference function"""
    # ONNX provider
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] \
                if config.get("device") == "cuda" else ['CPUExecutionProvider']
    # global calibration P2 matrix
    P2 = Calib.get_P(config.get("calib_file"))
    # dimension averages #TODO: depricated
    class_averages = ClassAverages()

    # time
    avg_time = {
        "initiate_detector": 0,
        "initiate_regressor": 0,
        "detector": 0,
        "regressor": 0,
        "plotting": 0,
    }

    # initialize detector model
    start_detector = time.time()
    log.info(f"Instantiating detector <{config.detector._target_}>")
    detector = hydra.utils.instantiate(config.detector)
    avg_time["initiate_detector"] = time.time() - start_detector

    # initialize regressor model
    start_regressor = time.time()
    if config.get("inference_type") == "pytorch":
        # pytorch regressor model
        log.info(f"Instantiating regressor <{config.model._target_}>")
        regressor: LightningModule = hydra.utils.instantiate(config.model)
        regressor.load_state_dict(torch.load(config.get("regressor_weights")))
        regressor.eval().to(config.get("device"))
    elif config.get("inference_type") == "onnx":
        # onnx regressor model
        log.info(f"Instantiating ONNX regressor <{config.get('regressor_weights').split('/')[-1]}>")
        regressor = onnxruntime.InferenceSession(config.get("regressor_weights"), providers=providers)
        input_name = regressor.get_inputs()[0].name
    elif config.get("inference_type") == "openvino":
        # openvino regressor model
        log.info(f"Instantiating OpenVINO regressor <{config.get('regressor_weights').split('/')[-1]}>")
        core = ov.Core()
        model = core.read_model(config.get("regressor_weights"))
        regressor = core.compile_model(model, 'CPU') #TODO: change to config.get("device")
        infer_req = regressor.create_infer_request()
    avg_time["initiate_regressor"] = time.time() - start_regressor

    # initialize preprocessing transforms
    log.info(f"Instantiating Preprocessing Transforms")
    preprocess: List[torch.nn.Module] = []
    if "augmentation" in config:
        for _, conf in config.augmentation.items():
            if "_target_" in conf:
                preprocess.append(hydra.utils.instantiate(conf))
    preprocess = transforms.Compose(preprocess)

    # Create output directory
    os.makedirs(config.get("output_dir"), exist_ok=True)

    # TODO: inference on video
    # loop thru images
    imgs_path = sorted(glob(os.path.join(config.get("source_dir"), "*")))
    for img_path in imgs_path:
        # Initialize object and plotting modules
        plot3dbev = Plot3DBoxBev(P2)

        img_name = img_path.split("/")[-1].split(".")[0]
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detect object with Detector
        start_detect = time.time()
        dets = detector(img).crop(save=config.get("save_det2d"))
        avg_time["detector"] += time.time() - start_detect

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

            start_reg = time.time()
            # regress 2D bbox with Regressor
            if config.get("inference_type") == "pytorch":
                [orient, conf, dim] = regressor(crop)
                orient = orient.cpu().detach().numpy()[0, :, :]
                conf = conf.cpu().detach().numpy()[0, :]
                dim = dim.cpu().detach().numpy()[0, :]
            elif config.get("inference_type") == "onnx":
                # TODO: inference with GPU
                [orient, conf, dim] = regressor.run(None, {input_name: crop.cpu().numpy()})
                orient = orient[0]
                conf = conf[0]
                dim = dim[0]
            elif config.get("inference_type") == "openvino":
                infer_req.infer(inputs={0: crop.cpu().numpy()})
                orient = infer_req.get_output_tensor(0).data[0]
                conf = infer_req.get_output_tensor(1).data[0]
                dim = infer_req.get_output_tensor(2).data[0]

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

            # output prediction label
            output_line = obj.member_to_list()
            output_line.append(1.0)
            output_line = " ".join([str(i) for i in output_line]) + "\n"

            avg_time["regressor"] += time.time() - start_reg

            # write results
            if config.get("save_txt"):
                with open(f"{config.get('output_dir')}/{img_name}.txt", "a") as f:
                    f.write(output_line)


            if config.get("save_result"):
                start_plot = time.time()
                plot3dbev.plot(
                    img=img,
                    class_object=obj.name.lower(),
                    bbox=[obj.xmin, obj.ymin, obj.xmax, obj.ymax],
                    dim=[obj.h, obj.w, obj.l],
                    loc=[obj.tx, obj.ty, obj.tz],
                    rot_y=obj.rot_global,
                )
                avg_time["plotting"] += time.time() - start_plot

        # save images
        if config.get("save_result"):
            # cv2.imwrite(f'{config.get("output_dir")}/{name}.png', img_draw)
            plot3dbev.save_plot(config.get("output_dir"), img_name)

    # print time
    for key, value in avg_time.items():
        if key in ["detector", "regressor", "plotting"]:
            avg_time[key] = value / len(imgs_path)
    log.info(f"Average Time: {avg_time}")

@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference_old(config: DictConfig):

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
            elif config.get("inference_type") == "onnx":
                # TODO: inference with GPU
                [orient, conf, dim] = regressor.run(None, {input_name: crop.cpu().numpy()})
                orient = orient[0]
                conf = conf[0]
                dim = dim[0]
            elif config.get("inference_type") == "openvino":
                infer_req.infer(inputs={0: crop.cpu().numpy()})
                orient = infer_req.get_output_tensor(0).data[0]
                conf = infer_req.get_output_tensor(1).data[0]
                dim = infer_req.get_output_tensor(2).data[0]

            # refinement dimension
            try:
                dim += class_averages.get_item(class_to_labels(det["cls"].cpu().numpy()))
                DIMS.append(dim)
            except:
                dim = DIMS[-1]
            # calculate orientation
            box = [box.cpu().numpy() for box in det["box"]]  # xyxy
            theta_ray = calc_theta_ray(img.size[0], box, proj_matrix)
            alpha = calc_alpha(orient=orient, conf=conf, bins=2)
            orient = alpha + theta_ray
            # calculate the location
            location, x = calc_location(
                dimension=dim,
                proj_matrix=proj_matrix,
                box_2d=box,
                alpha=alpha,
                theta_ray=theta_ray,
            )
            # plot 3d bbox
            plot_3d_box(
                img=img_draw,
                cam_to_img=proj_matrix,
                ry=orient,
                dimension=dim,
                center=location,
            )

            if config.get("save_txt"):
                # save txt results
                results_txt = {
                    "type": det["label"].split(" ")[0].capitalize(),
                    "truncated": "-1.00", # set default to -1.00
                    "occluded": -1,
                    "alpha": round(alpha, 2),
                    "bbox": " ".join(str(np.round(x, 2)) for x in box),
                    "dimension": " ".join(map(str, np.round(dim, 2))),
                    "location": " ".join(str(np.round(x, 2)) for x in location),
                    "rotation_y": round(orient, 2),
                    "score": str(np.round(det["conf"].cpu().numpy(), 2)),
                }
                # append as string
                RESULTS_TXT.append(" ".join(str(v) for k, v in results_txt.items()))

        # save images
        if config.get("save_result"):
            cv2.imwrite(f'{config.get("output_dir")}/{name}.png', img_draw)

        # save txt
        if config.get("save_txt"):
            with open(f'{config.get("output_dir")}/{name}.txt', "w") as f:
                for i in range(len(RESULTS_TXT)):
                    f.write(f"{RESULTS_TXT[i]}\n")

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
