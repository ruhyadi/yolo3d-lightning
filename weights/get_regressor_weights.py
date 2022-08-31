"""Get checkpoint from W&B"""

import wandb

run = wandb.init()
artifact = run.use_artifact('3ddetection/yolo3d-regressor/experiment-ckpts:v11', type='checkpoints')
artifact_dir = artifact.download()