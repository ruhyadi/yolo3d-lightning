""" Conver checkpoint to model (.pt/.pth/.onnx) """

import torch
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningModule
from src import utils

import dotenv
import hydra
from omegaconf import DictConfig
import os

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)
log = utils.get_pylogger(__name__)

@hydra.main(config_path="configs/", config_name="convert.yaml")
def convert(config: DictConfig):

    # assert model convertion
    assert config.get('convert_to') in ['pytorch', 'torchscript', 'onnx', 'tensorrt'], \
        "Please Choose one of [pytorch, torchscript, onnx, tensorrt]"

    # Init lightning model
    log.info(f"Instantiating model <{config.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Convert relative ckpt path to absolute path if necessary
    log.info(f"Load checkpoint <{config.get('checkpoint_dir')}>")
    ckpt_path = config.get("checkpoint_dir")
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = config.get(os.path.join(hydra.utils.get_original_cwd(), ckpt_path))

    # load model checkpoint
    model = model.load_from_checkpoint(ckpt_path)
    model.cuda()

    # input sample
    input_sample = config.get('input_sample')

    # Convert
    if config.get('convert_to') == 'pytorch':
        log.info("Convert to Pytorch (.pt)")
        torch.save(model.state_dict(), f'{config.get("name")}.pt')
        log.info(f"Saved model {config.get('name')}.pt")
    if config.get('convert_to') == 'torchscript':
        log.info("Convert to Torchscript (.pt)")
        torch.jit.save(model.to_torchscript(), f'{config.get("name")}.pt')
        log.info(f"Saved model {config.get('name')}.pt")
    if config.get('convert_to') == 'onnx':
        log.info("Convert to ONNX (.onnx)")
        model.cuda()
        input_sample = torch.rand((1, 3, 224, 224), device=torch.device('cuda'))
        model.to_onnx(f'{config.get("name")}.onnx', input_sample, export_params=True)
        log.info(f"Saved model {config.get('name')}.onnx")

if __name__ == '__main__':

    convert()