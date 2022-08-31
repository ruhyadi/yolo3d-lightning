# Quick Command

## Train Regressor Model

- Train original
```bash
python src/train.py
```

- With experiment
```bash
python src/train.py \
    experiment=sample
```

## Train Detector Model
### Yolov5

- Multi GPU Training
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

- Single GPU Training
```bash
cd yolov5
python train.py \
    --data ../configs/detector/yolov5_kitti.yaml \
    --weights yolov5s.pt \
    --img 640 
```

## Hyperparameter Tuning with Hydra

```bash
python src/train.py -m \
    hparams_search=regressor_optuna \
    experiment=sample_optuna
```