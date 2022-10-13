#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
from logging import Logger
import os
import yaml
import os.path as osp
from pathlib import Path
import torch
import torch.distributed as dist
import sys
import numpy as np 
import random
import time


ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    #sys.path.append(str(ROOT))

from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.metrics import fitness
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint, check_yaml, print_mutation, colorstr,plot_evolve


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)
    parser.add_argument('--weights', type=str, default=None, help='pretrained initial weights path')
    parser.add_argument('--data-path', default='./data/coco.yaml', type=str, help='path of dataset')
    parser.add_argument('--conf-file', default='./configs/yolov6n.py', type=str, help='experiments description file')
    parser.add_argument('--img-size', default=640, type=int, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', default=32, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--eval-interval', default=20, type=int, help='evaluate at every interval epochs')
    parser.add_argument('--eval-final-only', action='store_true', help='only evaluate at the final epoch')
    parser.add_argument('--heavy-eval-range', default=50, type=int,
                        help='evaluating every epoch for last such epochs (can be jointly used with --eval-interval)')

    parser.add_argument('--verbose', type=bool,default=True, help='eval cfg verbose print verbose eval info')
    parser.add_argument('--do_coco_metric', type=bool, default=True, help='eval cfg ')
    parser.add_argument('--do_pr_metric', type=bool, default=False, help='eval cfg do_pr_metric do_coco_metric')
    parser.add_argument('--plot_curve', type=bool, default=False, help='eval cfg plot_curve')
    parser.add_argument('--plot_confusion_matrix',type=bool, default=False, help='eval cfg plot_confusion_matrix')
    
    
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--name', default='exp', type=str, help='experiment name, saved to output_dir/name')
    parser.add_argument('--dist_url', default='env://', type=str, help='url used to set up distributed training')
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume the most recent training')
    parser.add_argument('--write_trainbatch_tb', action='store_true', help='write train_batch image to tensorboard once an epoch, may slightly slower train speed if open')
    parser.add_argument('--write_valimg_tb', action='store_true', help='write val images to tensorboard once an epoch, may slightly slower train speed  and super big tb logs if open')
    parser.add_argument('--stop_aug_last_n_epoch', default=15, type=int, help='stop strong aug at last n epoch, neg value not stop, default 15')
    parser.add_argument('--save_ckpt_on_last_n_epoch', default=-1, type=int, help='save last n epoch even not best or last, neg value not save')
    parser.add_argument('--quant', action='store_true', help='quant or not')
    parser.add_argument('--calib', action='store_true', help='run ptq')
    # 模型蒸馏参数
    parser.add_argument('--distill', action='store_true', help='distill or not')
    parser.add_argument('--distill_feat', action='store_true', help='distill featmap or not')
    parser.add_argument('--other_teacher_distill', action='store_true', help='self distill or distll student with other larger teachers')
    parser.add_argument('--teacher_conf_file', default='./configs/yolov6l.py', type=str, help='experiments description file for the teacher model')
    parser.add_argument('--teacher_hyp', type=str, default=None, help='hyperparameters path')
    parser.add_argument('--teacher_model_path', type=str, default=None, help='teacher model path')
    parser.add_argument('--temperature', type=int, default=5, help='distill temperature')
    
    parser.add_argument('--exist_ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--hyp', type=str, default=None, help='hyperparameters path')
    parser.add_argument('--evolve', type=int, nargs='?', const=300, help='evolve hyperparameters for x generations')
    parser.add_argument('--patience', type=int, default=100, help='evolve EarlyStopping patience (epochs without improvement)')
    return parser

def overwrite_cfg_hyps(cfg, hyp):
    for ow_k in ['data_aug', 'solver']:
        old_hyp = getattr(cfg, ow_k)
        for k, v in old_hyp.items():
            old_hyp[k] = hyp.get(k, v)
    return cfg
    
def check_and_init(args):
    '''check config files and device.'''
    # check files
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1

    if args.resume and not args.evolve:
        # args.resume can be a checkpoint file path or a boolean value.
        checkpoint_path = args.resume if isinstance(args.resume, str) else find_latest_checkpoint()
        assert os.path.isfile(checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
        LOGGER.info(f'Resume training from the checkpoint file :{checkpoint_path}')
        resume_opt_file_path = Path(checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))  # load args value from args.yaml
        else:
            LOGGER.warning(f'We can not find the path of {Path(checkpoint_path).parent.parent / "args.yaml"},'\
                           f' we will save exp log to {Path(checkpoint_path).parent.parent}')
            LOGGER.warning(f'In this case, make sure to provide configuration, such as data, batch size.')
            args.save_dir = str(Path(checkpoint_path).parent.parent, exist_ok=True)
        args.resume = checkpoint_path  # set the args.resume to checkpoint path.
    else:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name),exist_ok=args.exist_ok))
        if master_process:
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
    
    
    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')
        
    if args.other_teacher_distill:
        teacher_cfg = Config.fromfile(args.teacher_conf_file)
        if not hasattr(teacher_cfg, 'training_mode'):
            setattr(teacher_cfg, 'training_mode', 'repvgg')
    else:
        teacher_cfg = cfg
        
    # check device
    device = select_device(args.device)
    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))
    # save args
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return cfg, teacher_cfg, device, args


def main(args):
    '''main function of training'''
    # Setup
    args.rank, args.local_rank, args.world_size = get_envs()
    cfg, teacher_cfg, device, args = check_and_init(args)
    
    # reload envs because args was chagned in check_and_init(args)
    args.rank, args.local_rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')
    if args.local_rank != -1: # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size)
    
    if args.weights:
        cfg.model.pretrained = args.weights
        
    args.hyp = check_yaml(args.hyp)
    
    if args.teacher_hyp:
        with open(args.teacher_hyp, errors='ignore') as f:
            teacher_hyp = yaml.safe_load(f)  # load hyps dict
            teacher_cfg = overwrite_cfg_hyps(teacher_cfg, teacher_hyp.copy()) 
            teacher_cfg.model.head.use_dfl = True
            teacher_cfg.model.head.reg_max = 16
            
    if args.hyp:
        with open(args.hyp, errors='ignore') as f:
            hyp = yaml.safe_load(f)  # load hyps dict
            cfg = overwrite_cfg_hyps(cfg, hyp.copy())
            cfg.model.head.use_dfl = True
            cfg.model.head.reg_max = 16
          
    else: # default hyp
        hyp = dict(
            lr0=0.0032,
            lrf=0.12,
            momentum=0.843,
            weight_decay=0.00036,
            warmup_epochs=2.0,
            warmup_momentum=0.5,
            warmup_bias_lr=0.05,
            hsv_h=0.0138,
            hsv_s=0.664,
            hsv_v=0.464,
            degrees=0.373,
            translate=0.245,
            scale=0.898,
            shear=0.602,
            flipud=0.00856,
            fliplr=0.5,
            mosaic=1.0,
            mixup=0.243
        )
    
    # Start
    if not args.evolve:
        LOGGER.info('---------------------- cfg:-----------------------------')
        LOGGER.info(cfg)
        LOGGER.info('----------------------teacher cfg:-----------------------------')
        LOGGER.info(teacher_cfg)
        trainer = Trainer(args, cfg, device, teacher_cfg)

            # PTQ
        if args.quant and args.calib:
            trainer.calibrate(cfg)
            return
        results = trainer.train()
        # End
        if args.world_size > 1 and args.rank == 0:
            LOGGER.info('Destroying process group... ')
            dist.destroy_process_group()
    else:
        # Hyperparameter evolution metadata (mutation scale 0-1, lower_limit, upper_limit)
        meta = {
            'lr0': (1, 1e-5, 1e-1),  # initial learning rate (SGD=1E-2, Adam=1E-3)
            'lrf': (1, 0.01, 1.0),  # final OneCycleLR learning rate (lr0 * lrf)
            'momentum': (0.3, 0.6, 0.98),  # SGD momentum/Adam beta1
            'weight_decay': (1, 0.0, 0.001),  # optimizer weight decay
            'warmup_epochs': (1, 0.0, 5.0),  # warmup epochs (fractions ok)
            'warmup_momentum': (1, 0.0, 0.95),  # warmup initial momentum
            'warmup_bias_lr': (1, 0.0, 0.2),  # warmup initial bias lr
            'hsv_h': (1, 0.0, 0.1),  # image HSV-Hue augmentation (fraction)
            'hsv_s': (1, 0.0, 0.9),  # image HSV-Saturation augmentation (fraction)
            'hsv_v': (1, 0.0, 0.9),  # image HSV-Value augmentation (fraction)
            'degrees': (1, 0.0, 45.0),  # image rotation (+/- deg)
            'translate': (1, 0.0, 0.9),  # image translation (+/- fraction)
            'scale': (1, 0.0, 0.9),  # image scale (+/- gain)
            'shear': (1, 0.0, 10.0),  # image shear (+/- deg)
            'flipud': (1, 0.0, 1.0),  # image flip up-down (probability)
            'fliplr': (0, 0.0, 1.0),  # image flip left-right (probability)
            'mosaic': (1, 0.0, 1.0),  # image mixup (probability)
            'mixup': (1, 0.0, 1.0),  # image mixup (probability)
        }
        evolve_yaml, evolve_csv = os.path.join(args.save_dir,'hyp_evolve.yaml'), os.path.join(args.save_dir, 'evolve.csv')

        for _ in range(args.evolve):  # generations to evolve
            if os.path.exists(evolve_csv):  # if evolve.csv exists: select best hyps and mutate
                # Select parent(s)
                parent = 'weighted'  # parent selection method: 'single' or 'weighted'
                x = np.loadtxt(evolve_csv, ndmin=2, delimiter=',', skiprows=1)
                n = min(5, len(x))  # number of previous results to consider
                x = x[np.argsort(-fitness(x))][:n]  # top n mutations
                w = fitness(x) - fitness(x).min() + 1E-6  # weights (sum > 0)
                if parent == 'single' or len(x) == 1:
                    # x = x[random.randint(0, n - 1)]  # random selection
                    x = x[random.choices(range(n), weights=w)[0]]  # weighted selection
                elif parent == 'weighted':
                    x = (x * w.reshape(n, 1)).sum(0) / w.sum()  # weighted combination

                # Mutate
                mp, s = 0.8, 0.2  # mutation probability, sigma
                npr = np.random
                npr.seed(int(time.time()))
                g = np.array([meta[k][0] for k in hyp.keys()])  # gains 0-1
                ng = len(meta)
                v = np.ones(ng)
                while all(v == 1):  # mutate until a change occurs (prevent duplicates)
                    v = (g * (npr.random(ng) < mp) * npr.randn(ng) * npr.random() * s + 1).clip(0.3, 3.0)
                for i, k in enumerate(hyp.keys()):  # plt.hist(v.ravel(), 300)
                    hyp[k] = float(x[i + 4] * v[i])  # mutate
            
   
            # Constrain to limits
            for k, v in meta.items():
                hyp[k] = max(hyp[k], v[1])  # lower limit
                hyp[k] = min(hyp[k], v[2])  # upper limit
                hyp[k] = round(hyp[k], 5)  # significant digits
                
            # overwrite hyp
            cfg = overwrite_cfg_hyps(cfg, hyp.copy())

            # Train mutation
            trainer = Trainer(args, cfg, device)
            results = trainer.train()
            #results = ('metrics/precision', 'metrics/recall', 'metrics/mAP_0.5', 'metrics/mAP_0.5:0.95')
            # Write mutation results
            print_mutation(results, hyp.copy(), args.save_dir)

        # Plot results
        plot_evolve(evolve_csv)
        LOGGER.info(f'Hyperparameter evolution finished {args.evolve} generations\n'
                    f"Results saved to {colorstr('bold', args.save_dir)}\n"
                    f'Usage example: $ python train.py --hyp {evolve_yaml}')



if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
