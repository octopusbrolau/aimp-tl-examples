#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil
# import torch
# import warnings

def set_logging(name=None):
    rank = int(os.getenv('RANK', -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


LOGGER = set_logging(__name__)
NCOLS = shutil.get_terminal_size().columns

# def is_parallel(model):
#     # Returns True if model is of type DP or DDP
#     return type(model) in (torch.nn.parallel.DataParallel, torch.nn.parallel.DistributedDataParallel)

# def de_parallel(model):
#     # De-parallelize a model: returns single-GPU model if model is of type DP or DDP
#     return model.module if is_parallel(model) else model

def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors='ignore') as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, 'w') as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("metrics/mAP_0.5", results[0], epoch + 1)
    tblogger.add_scalar("metrics/mAP_0.5:0.95", results[1], epoch + 1)
    tblogger.add_scalar("metrics/precision", results[2], epoch + 1)
    tblogger.add_scalar("metrics/recall", results[3], epoch + 1)
    #tblogger.add_scalar("val/F1", results[4], epoch + 1)
    
    tblogger.add_scalar("train/box_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)

    tblogger.add_scalar("x/lr0", results[5], epoch + 1)
    tblogger.add_scalar("x/lr1", results[6], epoch + 1)
    tblogger.add_scalar("x/lr2", results[7], epoch + 1)

    
def write_tbimg(tblogger, imgs, step, type='train'):
    """Display train_batch and validation predictions to tensorboard."""
    if type == 'train':
        tblogger.add_image(f'train_batch', imgs, step + 1, dataformats='HWC')
    elif type == 'val':
        for idx, img in enumerate(imgs):
            tblogger.add_image(f'val_img_{idx + 1}', img, step + 1, dataformats='HWC')
    else:
        LOGGER.warning('WARNING: Unknown image type to visualize.\n')
        

# def write_tensorboard_graph(tblogger, model, imgsz=(640, 640)):
#     # Log model graph to TensorBoard
#     try:
#         p = next(model.parameters())  # for device, type
#         imgsz = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz  # expand
#         im = torch.zeros((1, 3, *imgsz)).to(p.device).type_as(p)  # input image (WARNING: must be zeros, not empty)
#         with warnings.catch_warnings():
#             warnings.simplefilter('ignore')  # suppress jit trace warning
#             tblogger.add_graph(torch.jit.trace(de_parallel(model), im, strict=False), [])
#     except Exception as e:
#         LOGGER.warning(f'WARNING ⚠️ TensorBoard graph visualization failure {e}')
 
