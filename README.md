<div align="center">

# YOLO3D: 3D Object Detection with YOLO

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8+-blue?style=flat&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=flat&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=flat&logo=pytorchlightning&logoColor=white"></a>

<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=flat&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=flat&labelColor=gray"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

## ⚠️&nbsp;&nbsp;Cautions
> This repository currently under development

## 📼&nbsp;&nbsp;Demo
<div align="center">

![demo](./docs/assets/demo.gif)

</div>

## 📌&nbsp;&nbsp;Introduction

YOLO3D is inspired by [Mousavian et al.](https://arxiv.org/abs/1612.00496) in their paper **3D Bounding Box Estimation Using Deep Learning and Geometry**. YOLO3D uses a different approach, as the detector uses [YOLOv5](https://github.com/ultralytics/yolov5) which previously used Faster-RCNN, and Regressor uses ResNet18/VGG11 which was previously VGG19.

## 🚀&nbsp;&nbsp;Quickstart
> YOLO3D use hydra as the config manager; please follow [official website](https://hydra.cc/) or [ashleve/lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template).

### 🍿&nbsp;&nbsp;Inference
You can use pretrained weight from [Release](https://github.com/ruhyadi/yolo3d-lightning/releases), you can download it using script `get_weights.py`:
```bash
# download pretrained model
python script/get_weights.py \
  --tag v0.1 \
  --dir ./weights
```
Inference with `inference.py`:
```bash
python inference.py \
  source_dir="./data/demo/images" \
  detector.model_path="./weights/detector_yolov5s.pt" \
  regressor_weights="./weights/regressor_resnet18.pt"
```

## ⚔️ Training
There are two models that will be trained here: **detector** and **regressor**. For now, the detector model that can be used is only **YOLOv5**, while the regressor model can use all models supported by **Torchvision**.

### 💽 Dataset Preparation
For now, YOLO3D only supports the [KITTI dataset](http://www.cvlibs.net/datasets/kitti/). Going forward, we will try to add support to the Lyft and nuScene datasets.
#### 1. Download KITTI Dataset
You can download KITTI dataset from [official website](http://www.cvlibs.net/datasets/kitti/). After that, extract dataset to `data/KITTI`. Since we will be using two models, it is highly recommended to rename `images_2` to `images`.

```bash
.
├── data
│   └── KITTI
│       ├── calib
│       ├── images # original images_2
│       └── labels_2
```

#### 2. Generate YOLO Labels

The kitti label format on `labels` is different from the format required by the YOLO model. Therefore, we have to create a YOLO format from a KITTI format. The author has provided a `script/kitti_to_yolo.py` that can be used.

```bash
python script/kitti_to_yolo.py \
  --dataset_path ./data/KITTI \
  --classes car, van, truck, pedestrian, cyclist \
  --img_width 1224 \
  --img_height 370
```
The script will generate a `labels` folder containing the labels for each image in YOLO format.

```bash
.
├── data
│   └── KITTI
│       ├── calib
│       ├── images    # original images_2
|       ├── labels_2  # kitti labels
│       └── labels    # yolo labels
```

The next thing is to generate a sets of images/labels training and validation, these sets are also used as partitions to divide the dataset. The author has provided a `script/generate_sets.py` that can be used.

```bash
python script/generate_sets.py \
  --images_path ./data/KITTI/images \
  --dump_dir ./data/KITTI \
  --postfix _yolo \
  --train_size 0.8 \
  --is_yolo
```

### 🚀 Training Detector Model
> Right now author just use YOLOv5 model

For YOLOv5 training on a **single GPU**, you can use the command below:

```bash
cd yolov5
python train.py \
    --data ../configs/detector/yolov5_kitti.yaml \
    --weights yolov5s.pt \
    --img 640 
```

As for training on **multiple GPUs**, you can use the command below:

```bash
cd yolov5
python -m torch.distributed.launch \
    --nproc_per_node 4 train.py \
    --epochs 10 \
    --batch 64 \
    --data ../configs/detector/yolov5_kitti.yaml \
    --weights yolov5s.pt \
    --device 0,1,2,3
```

### 🪀 Training Regessor Model
> ⚠️ Under development

You can use all the models available on **Torchvision** by adding some configuration to `src/models/components/base.py`. The current author has provided **ResNet18** and **VGG11** which can be used directly.

```bash
python src/train.py \
  experiment=sample
```

## ❤️&nbsp;&nbsp;Acknowledgement

- [YOLOv5 by Ultralytics](https://github.com/ultralytics/yolov5)
- [skhadem/3D-BoundingBox](https://github.com/skhadem/3D-BoundingBox)
- [Mousavian et al.](https://arxiv.org/abs/1612.00496)
```
@misc{mousavian20173d,
      title={3D Bounding Box Estimation Using Deep Learning and Geometry}, 
      author={Arsalan Mousavian and Dragomir Anguelov and John Flynn and Jana Kosecka},
      year={2017},
      eprint={1612.00496},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```