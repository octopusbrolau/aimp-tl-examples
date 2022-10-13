#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import glob
import torch
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import matplotlib
matplotlib.rc('font', **{'size': 11})
matplotlib.use('Agg')  # for writing to files only
import matplotlib.pyplot as plt

def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, (xedges, yedges))
    xidx = np.clip(np.digitize(x, xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])

def plot_evolve(evolve_csv='path/to/evolve.csv'):  # from utils.plots import *; plot_evolve()
    # Plot evolve.csv hyp evolution results
    evolve_csv = Path(evolve_csv)
    data = pd.read_csv(evolve_csv)
    keys = [x.strip() for x in data.columns]
    x = data.values
    f = fitness(x)
    j = np.argmax(f)  # max fitness index
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    print(f'Best results from row {j} of {evolve_csv}:')
    for i, k in enumerate(keys[7:]):
        v = x[:, 7 + i]
        mu = v[j]  # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(v, f, c=hist2d(v, f, 20), cmap='viridis', alpha=.8, edgecolors='none')
        plt.plot(mu, f.max(), 'k+', markersize=15)
        plt.title(f'{k} = {mu:.3g}', fontdict={'size': 9})  # limit to 40 characters
        if i % 5 != 0:
            plt.yticks([])
        print(f'{k:>15}: {mu:.3g}')
    f = evolve_csv.with_suffix('.png')  # filename
    plt.savefig(f, dpi=200)
    plt.close()
    print(f'Saved {f}')
    
def increment_name(path, exist_ok=False):
    '''increase save directory's id'''
    path = Path(path)
    sep = ''
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(''), path.suffix) if path.is_file() else (path, '')
        for n in range(1, 9999):
            p = f'{path}{sep}{n}{suffix}'
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


def find_latest_checkpoint(search_dir='.'):
    '''Find the most recent saved checkpoint in search_dir.'''
    checkpoint_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ''


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def xywh2xyxy(bboxes):
    '''Transform bbox(xywh) to box(xyxy).'''
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5 
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)

def check_suffix(file='yolov5s.pt', suffix=('.pt',), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"


def check_yaml(file, suffix=('.yaml', '.yml')):
    # Search/download YAML file (if necessary) and return path, checking suffix
    return check_file(file, suffix)

def check_file(file, suffix=''):
    # Search/download file (if necessary) and return path
    check_suffix(file, suffix)  # optional
    file = str(file)  # convert to str()
    if Path(file).is_file() or not file:  # exists
        return file

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {
        'black': '\033[30m',  # basic colors
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # bright colors
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # misc
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']

def fitness(x):
    # Model fitness as a weighted combination of metrics
    w = [0.0, 0.0, 0.8, 0.2]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]
    return (x[:, :4] * w).sum(1)

def print_mutation(results, hyp, save_dir, prefix=colorstr('evolve: ')):
    evolve_csv = os.path.join(save_dir, 'evolve.csv')
    evolve_yaml = os.path.join(save_dir, 'hyp_evolve.yaml')
    keys = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95') + tuple(hyp.keys())  # [results + hyps]
    keys = tuple(x.strip() for x in keys)
    vals = results + tuple(hyp.values())
    n = len(keys)


    # Log to evolve.csv
    s = '' if os.path.exists(evolve_csv) else (('%20s,' * n % keys).rstrip(',') + '\n')  # add header
    with open(evolve_csv, 'a') as f:
        f.write(s + ('%20.5g,' * n % vals).rstrip(',') + '\n')

    # Save yaml
    with open(evolve_yaml, 'w') as f:
        data = pd.read_csv(evolve_csv)
        data = data.rename(columns=lambda x: x.strip())  # strip keys
        i = np.argmax(fitness(data.values[:, :4]))  #
        generations = len(data)
        f.write('# YOLOv5 Hyperparameter Evolution Results\n' + f'# Best generation: {i}\n' +
                f'# Last generation: {generations - 1}\n' + '# ' + ', '.join(f'{x.strip():>20s}' for x in keys[:4]) +
                '\n' + '# ' + ', '.join(f'{x:>20.5g}' for x in data.values[i, :4]) + '\n\n')
        yaml.safe_dump(data.loc[i][4:].to_dict(), f, sort_keys=False)

    # Print to screen
    print(prefix + f'{generations} generations finished, current result:\n' + prefix +
                ', '.join(f'{x.strip():>20s}' for x in keys) + '\n' + prefix + ', '.join(f'{x:20.5g}'
                                                                                         for x in vals) + '\n\n')
