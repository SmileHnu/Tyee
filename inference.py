#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
Inference launcher for PRL/tyee.

Usage example:
    python inference.py -c config.yaml --checkpoint path/to/checkpoint.pt --split test --fold 0 --output out.npz

This script mirrors parts of `tyee/main.py` but runs a single-process inference using Trainer.infer_fold().
"""
import os
import yaml
import torch
import random
import datetime
import numpy as np
from argparse import ArgumentParser
from collections import OrderedDict

from tyee.trainer import Trainer
from tyee.utils.log_utils import create_experiment_directories, init_logging, save_config
from tyee.utils.cfg_utils import get_nested_field, convert_sci_notation
from tyee.utils.import_utils import import_user_module


class InferenceTrainer(Trainer):
    """
    A subclass of Trainer specifically for inference.
    It overrides checkpoint loading to be more robust to DDP prefixes and 
    preserves the inference configuration.
    """
    def _load_checkpoint(self, filename):
        self.logger.info(f"Loading checkpoint from {filename}")
        # Load to CPU first to avoid OOM, then move to device
        ckpt_params = torch.load(filename, map_location='cpu')
        
        # 1. Handle DDP 'module.' prefix
        state_dict = ckpt_params["model"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k  # remove 'module.'
            new_state_dict[name] = v
        
        # 2. Load state dict
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        
        if missing_keys:
            self.logger.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            self.logger.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")

        # 3. Do NOT overwrite self.cfg with ckpt_params["args"]
        # We want to use the config provided at inference time (e.g. for dataset paths),
        # not the one stored in the checkpoint.
        
        if "best_metric_value" in ckpt_params:
            self.best_metric_value = ckpt_params["best_metric_value"]
            
        self.logger.info(f"Model loaded successfully. Best metric from training: {self.best_metric_value}")


def update_config(cfg, key_list, value):
    """Helper to update nested config dictionary."""
    sub_cfg = cfg
    for key in key_list[:-1]:
        if key not in sub_cfg:
            sub_cfg[key] = {}
        sub_cfg = sub_cfg[key]
    
    # Try to convert value to appropriate type
    try:
        if value.lower() == 'true': value = True
        elif value.lower() == 'false': value = False
        elif value.isdigit(): value = int(value)
        else:
            try:
                value = float(value)
            except ValueError:
                pass
    except AttributeError:
        pass
        
    sub_cfg[key_list[-1]] = value


def load_cfg():
    parser = ArgumentParser(description="tyee inference launcher")
    parser.add_argument('-c', '--config', required=True, help='YAML config file')
    parser.add_argument('--checkpoint', required=False, help='Checkpoint path to load')
    parser.add_argument('--fold', type=int, default=0, help='Fold index to run inference on')
    parser.add_argument('--split', type=str, default='test', help="Which split to run: 'test' or 'val'")
    parser.add_argument('--output', type=str, default=None, help='Output file path (npz)')
    parser.add_argument('--batch_size', type=int, default=None, help='Override batch size for inference')
    parser.add_argument('--override', nargs='*', help='override cfg entries key=value (e.g. dataset.batch_size=16)', default=None)

    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file {args.config} not found")

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Handle overrides
    if args.override:
        for item in args.override:
            if '=' in item:
                k, v = item.split('=', 1)
                keys = k.split('.')
                update_config(cfg, keys, v)
                print(f"Overriding config: {k} = {v}")

    # Handle explicit batch size override
    if args.batch_size is not None:
        update_config(cfg, ['dataset', 'batch_size'], args.batch_size)
        print(f"Overriding batch size: {args.batch_size}")

    cfg = convert_sci_notation(cfg)

    seed = get_nested_field(cfg, 'common.seed', 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    return cfg, args


def main():
    cfg, args = load_cfg()

    # import user modules if any
    user_dir = get_nested_field(cfg, 'common.user_dir', None)
    import_user_module(user_dir)

    # prepare simple experiment folder so checkpoints/logs behave same as training
    root = get_nested_field(cfg, 'common.root', './experiments/')
    task_select = get_nested_field(cfg, 'task.select', 'task')
    
    # Use a dedicated inference directory or timestamp
    exp_dir = f"{root}/inference/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}-{task_select}"
    tb_dir, checkpoint_dir = create_experiment_directories(exp_dir)
    init_logging(exp_dir)
    # save_config(cfg, exp_dir) # Optional: save inference config

    cfg['common'] = cfg.get('common', {})
    cfg['common']['exp_dir'] = exp_dir
    cfg['common']['tb_dir'] = tb_dir
    cfg['common']['checkpoint_dir'] = checkpoint_dir

    # Use InferenceTrainer instead of standard Trainer
    trainer = InferenceTrainer(cfg, rank=0, world_size=1)

    metrics, out_path = trainer.infer_fold(fold=args.fold, checkpoint=args.checkpoint, split=args.split, output_file=args.output)

    print(f"Inference finished.\nMetrics: {metrics}\nSaved outputs: {out_path}")


if __name__ == '__main__':
    main()
