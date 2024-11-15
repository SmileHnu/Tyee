#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : trainer.py
@Time    : 2024/09/25 16:46:17
@Desc    : 
"""
import os
import torch
import datetime
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import lazy_import_module, get_nested_field, MetricEvaluator


class Trainer(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # 分布式环境配置
        self._ddp_backend = get_nested_field(self.cfg, "trainer.ddp_backend", "nccl")
        self._init_method = get_nested_field(self.cfg, "trainer.init_method", "tcp://127.0.0.1:60575")

        # 自动混合精度配置
        self.fp16 = get_nested_field(cfg, 'trainer.fp16', False)

        # 训练配置
        self._total_steps = get_nested_field(cfg, 'trainer.total_steps', 100)
        self._save_interval = get_nested_field(cfg, 'trainer.save_interval', 10)
        self._eval_interval = get_nested_field(cfg, 'trainer.eval_interval', 10)
        self._log_interval = get_nested_field(cfg, 'trainer.log_interval', 10)
        self._batch_size = get_nested_field(cfg, 'dataset.batch_size', 1)

        # 指标
        self.eval_metric = get_nested_field(cfg, 'trainer.eval_metric', None)
        metrics_dict = get_nested_field(cfg, 'trainer.metrics', None)
        self._metrics_evaluator = MetricEvaluator(metrics_dict)
        

        # 保存路径
        root = get_nested_field(cfg, "common.exp_dir", "./experiments/")
        # 任务配置
        self.task_select = get_nested_field(cfg, 'task.select', '')
        
        # 任务
        self.task = self._build_task()


    def _build_task(self) -> object:
        task = lazy_import_module('tasks', self.task_select)
        return task(self.cfg)
    
    def _distributed_init(self, world_size: int, rank: int) -> None:
        """
        initialize distributed environment
        :param int world_size: the number of total processes
        :param int rank: the current process index
        """
        # 初始化分布式环境
        dist.init_process_group(
            backend=self._ddp_backend,
            init_method=self._init_method,
            world_size=world_size,
            rank=rank,
        )
 
    def train(self, rank, world_size):
        """
        训练模型的主函数，执行训练过程，包括训练、评估、日志记录和模型保存等。

        :param rank: int, 当前进程的rank，用于分布式训练中标识进程。
        :param world_size: int, 总进程数，用于分布式训练中的参数设置。
        """
        # 初始化分布式进程
        # self.distributed_initializer(rank=rank, world_size=world_size)

        # 加载数据集
        train_dataset = self.task.get_train_dataset()
        dev_dataset = self.task.get_dev_dataset()
        test_dataset = self.task.get_test_dataset()

        # 模型、优化器与学习率调度器
        model = self.task.build_model()
        optimizer = self.task.build_optimizer(model)
        lr_scheduler = self.task.build_lr_scheduler(optimizer)

        device = torch.device(rank)
        model = model.to(device)

        # 构造采样器
        sampler = self.task.build_sampler(train_dataset, world_size, rank)

        # 构造数据加载器
        train_loader = self.task.build_dataloader(train_dataset, self._batch_size, sampler=sampler)
        dev_loader = self.task.build_dataloader(dev_dataset, self._batch_size, sampler=None)
        test_loader = self.task.build_dataloader(test_dataset, self._batch_size, sampler=None)

        # 如果使用FP16，初始化 GradScaler
        scaler = GradScaler() if self.fp16 else None

        iterator = self.task.get_batch_iterator(train_loader)
        for step in range(self._total_steps):
            # 更新epoch和迭代器
            if step % len(train_loader) == 0:
                sampler.set_epoch(step // len(train_loader))
                iterator = self.task.get_batch_iterator(train_loader)
            # train
            tot_loss = 0
            tot_loss += self._train_step(iterator, model, optimizer, lr_scheduler, scaler, device)

            # eval
            if step % self._eval_interval == 0:
                # 验证集
                dev_metrics = self._eval_step(model, dev_loader, device, world_size)

                # 保存最好的模型
                pass

                print(f"Step {step}, ", end="")
                # 动态打印所有在 metrics 中的指标
                metrics_output = ", ".join([f"{metric}: {value:.4f}" for metric, value in dev_metrics.items()])
                print(f"{metrics_output}")

                # 训练集
                test_metrics = self._eval_step(model, test_loader, device, world_size)
                print(f"Step {step}, ", end="")
                # 动态打印所有在 metrics 中的指标
                metrics_output = ", ".join([f"{metric}: {value:.4f}" for metric, value in test_metrics.items()])
                print(f"{metrics_output}")

            # log
            if step % self._log_interval == 0:
                print(f"Step {step}, Loss: {tot_loss / self._log_interval}")
                tot_loss = 0

                metrics = self._eval_step(model, test_loader, device, world_size)

                # acc = (correct / total).item()
                # print(f"Step {step}, Test Loss: {tot_eval_loss / total}, acc: {acc}")
                print(f"Step {step}, ", end="")
                # 动态打印所有在 metrics 中的指标
                metrics_output = ", ".join([f"{metric}: {value:.4f}" for metric, value in metrics.items()])
                print(f"{metrics_output}")
            
            # save
            if step % self._save_interval == 0:
                pass

        # 结束分布式进程
        # dist.destroy_process_group()

    
    def _train_step(self, iterator, model, optimizer, lr_scheduler, scaler, device):
        """
        执行单个训练步骤，包括前向传播、损失计算、反向传播和参数更新。

        :param iterator: iterator, 用于获取当前批次数据的迭代器。
        :param model: torch.nn.Module, 当前模型。
        :param optimizer: torch.optim.Optimizer, 用于优化模型参数的优化器。
        :param lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 学习率调度器，用于调整学习率。
        :param scaler: torch.cuda.amp.GradScaler, 用于混合精度训练时的梯度缩放。
        :param device: torch.device, 模型和数据的计算设备（如GPU）。
        :return: float, 当前训练步骤的损失值。
        """
        sample = self.task.load_sample(iterator).to(device)
        model.train()
        optimizer.zero_grad()
        # 使用自动混合精度（autocast）
        with autocast(enabled=self.fp16):
            result = self.task.train_step(model, sample)
            loss = result['loss']
            

        # 如果启用FP16，使用Scaler来缩放梯度
        if self.fp16:
            scaler.scale(loss).backward()  # 使用缩放后的梯度反向传播
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新Scaler状态
        else:
            loss.backward()  # 不使用FP16，常规反向传播
            optimizer.step()  # 更新参数
        
        if lr_scheduler:
            lr_scheduler.step()

        return loss.item()

    def _eval_step(self, model, loader, device, world_size):
        """
        执行单个评估步骤，包括模型评估、损失计算以及指标计算。

        :param model: torch.nn.Module, 当前模型。
        :param loader: DataLoader, 用于加载数据的迭代器。
        :param device: torch.device, 模型和数据的计算设备（如GPU）。
        :param world_size: int, 分布式训练中的总进程数，用于计算全局的平均损失。
        :return: tuple, 包含总评估损失和计算出的指标字典。
        """

        model.eval()
        with torch.no_grad():
            total, correct = 0, 0
            tot_eval_loss = 0
            for sample in self.task.get_batch_iterator(loader):
                for k, v in sample.items():
                    if v is not None:
                        sample[k] = v.to(device)
                result = self.task.valid_step(model, sample)
                tot_eval_loss += result['loss'].item()  # 累加损失
                total += len(result["target"])

                self._metrics_evaluator.update_metrics(result)  # 更新指标

            metrics = self._metrics_evaluator.calculate_metrics()  # 计算最终的指标

        # 同步损失值，确保所有进程计算相同的损失
        tot_eval_loss = torch.tensor(tot_eval_loss).to(device)
        dist.all_reduce(tot_eval_loss, op=dist.ReduceOp.SUM)  # 聚合所有进程的损失
        tot_eval_loss /= world_size  # 计算所有进程的平均损失

        # 同步所有指标，确保所有进程计算相同的指标
        for metric, value in metrics.items():
            if isinstance(value, torch.Tensor):  # 仅同步 tensor 类型的指标
                dist.all_reduce(value, op=dist.ReduceOp.SUM)  # 聚合所有进程的指标
                value /= world_size  # 计算所有进程的平均值

        # 将所有指标转化为标量并返回
        metrics = {metric: value.item() for metric, value in metrics.items()}

        return tot_eval_loss.item(), metrics

        


    def _state_dict(self):
        return {
            "args": self.cfg,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict()
        }
    
    def _save_checkpoint(self, filename):
        checkpoint = self.state_dict()
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def _load_checkpoint(self, filename):
        ckpt_params = torch.load(filename)
        self.args = ckpt_params["args"]
        self.model.load_state_dict(ckpt_params["model"])
        self.optimizer.load_state_dict(ckpt_params["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt_params["lr_scheduler"])

    
        
