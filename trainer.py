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
import logging
import torch.distributed as dist
from utils import MetricEvaluator
from utils import log_utils
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import lazy_import_module, get_nested_field, get_grad_norm
from torch.nn.parallel import DistributedDataParallel as DDP



class Trainer(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # 获取任务名称
        self.task_select = get_nested_field(cfg, 'task.select', '')
        
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
        
        # 任务
        self.task = self._build_task()
        self._metrics = get_nested_field(cfg, 'trainer.metrics', ['accuracy_score'])

        # 从配置中读取实验保存路径
        self.exp_dir = get_nested_field(self.cfg, "common.exp_dir")
        self.tb_dir = get_nested_field(self.cfg, "common.tb_dir")
        self.checkpoint_dir = get_nested_field(self.cfg, "common.checkpoint_dir")

        # 配置logging
        self.logger = logging.getLogger(__name__)

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=self.tb_dir)


    def _build_task(self) -> object:
        task = lazy_import_module('tasks', self.task_select)
        return task(self.cfg)
    
 
    def train(self, rank, world_size):
        """
        训练模型的主函数，执行训练过程，包括训练、评估、日志记录和模型保存等。

        :param rank: int, 当前进程的rank，用于分布式训练中标识进程。
        :param world_size: int, 总进程数，用于分布式训练中的参数设置。
        """
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
        if world_size > 1:
            model = DDP(model, device_ids=[rank])

        # 构造采样器
        train_sampler = self.task.build_sampler(train_dataset, world_size, rank)
        if dev_dataset is not None:
            dev_sampler = self.task.build_sampler(dev_dataset, world_size, rank)
        else:
            dev_sampler = None
        if test_dataset is not None:
            test_sampler = self.task.build_sampler(test_dataset, world_size, rank)
        else:
            test_sampler = None
        

        # 构造数据加载器
        train_loader = self.task.build_dataloader(train_dataset, self._batch_size, sampler=train_sampler, shuffle=True)
        if dev_dataset is not None:
            dev_loader = self.task.build_dataloader(dev_dataset, self._batch_size, sampler=dev_sampler, shuffle=False)
        else:
            dev_loader = None
        if test_dataset is not None:
            test_loader = self.task.build_dataloader(test_dataset, self._batch_size, sampler=test_sampler, shuffle=False)
        else:
            test_loader = None

        # 如果使用FP16，初始化 GradScaler
        scaler = torch.amp.GradScaler(enabled=self.fp16) if self.fp16 else None


        iterator = self.task.get_batch_iterator(train_loader)
        # 初始化累积指标
        accumulated_train_metrics = {}

        for step in range(1, self._total_steps + 1):
            # 更新epoch和迭代器
            if step % len(train_loader) == 0:
                train_sampler.set_epoch(step // len(train_loader))
                iterator = self.task.get_batch_iterator(train_loader)
             # 训练步骤
            train_result = self._train_step(iterator, model, optimizer, lr_scheduler, scaler, device, world_size)

            # 累积训练指标
            for metric, value in train_result.items():
                if metric not in accumulated_train_metrics:
                    accumulated_train_metrics[metric] = 0
                accumulated_train_metrics[metric] += value

            # 在更新参数之前，计算梯度范数
            grad_norm = get_grad_norm(model.parameters(), norm_type=2)  # 可以选择L2范数

            # 打印lr
            if rank == 0:
                min_lr = 10.
                max_lr = 0.
                for group in optimizer.param_groups:
                    min_lr = min(min_lr, group["lr"])
                    max_lr = max(max_lr, group["lr"])
                    weight_decay = group['weight_decay']

                self.writer.add_scalar(f'opt/lr', max_lr, step)
                self.writer.add_scalar(f'opt/min_lr', min_lr, step)
                self.writer.add_scalar(f'opt/weight_decay', weight_decay, step)
                self.writer.add_scalar(f'opt/grad_norm', grad_norm, step)
            # 日志记录
            if step % self._log_interval == 0 and rank == 0:
                # 计算平均训练指标
                avg_train_metrics = {
                    metric: round(value.item() / self._log_interval, 4) if isinstance(value, torch.Tensor) else round(value / self._log_interval, 4)
                    for metric, value in accumulated_train_metrics.items()
                }
            
                # 打印训练日志
                self.logger.info(f"Train, Step {step}, Metrics: {avg_train_metrics}")
                # 重置累积值
                accumulated_train_metrics = {}
                # 写入 TensorBoard
                for metric, value in avg_train_metrics.items():
                    self.writer.add_scalar(f'train/{metric}', value, step)
                

            # 验证集日志
            if step % self._eval_interval == 0:
                
                # 验证集评估
                dev_result = self._eval_step(model, dev_loader, device, world_size)
                # 测试集评估
                if test_loader is not None:
                    test_result = self._eval_step(model, test_loader, device, world_size)

                if rank == 0:
                    # 打印验证日志
                    self.logger.info(f"Dev, Step {step}, Metrics: {dev_result}")

                    # 写入 TensorBoard
                    for metric, value in dev_result.items():
                        self.writer.add_scalar(f'dev/{metric}', value, step)

                    if test_loader is not None:
                        
                        # 打印测试日志
                        self.logger.info(f"Test, Step {step}, Metrics: {test_result}")

                        for metric, value in test_result.items():
                            self.writer.add_scalar(f'test/{metric}', value, step)

                    # 判断评估指标是否达到了最优
                    current_metric_value = dev_result.get(self.eval_metric, None)  # 获取当前评估指标的值

                    # 初始化最优指标值和模型保存路径
                    if current_metric_value is not None:
                        save_best = False

                        # 如果已经有 best_metric_value，比较当前指标是否更优
                        if hasattr(self, 'best_metric_value'):
                            if current_metric_value > self.best_metric_value:
                                save_best = True
                        else:
                            # 如果没有 best_metric_value，初始化为当前指标
                            save_best = True

                        if save_best:
                            self.best_metric_value = current_metric_value
                            best_model_filename = f"{self.checkpoint_dir}/checkpoint_best.pt"
                            self._save_checkpoint(best_model_filename, model, optimizer, lr_scheduler)
                            self.logger.info(f"Best checkpoint saved with {self.eval_metric}: {current_metric_value}")


            if step % self._save_interval == 0 and rank == 0:
                # 构造保存路径和文件名
                checkpoint_filename = f"{self.checkpoint_dir}/checkpoint_step_{step}.pt"

                # 保存模型参数
                self._save_checkpoint(checkpoint_filename, model, optimizer, lr_scheduler)

                # 删除之前的检查点文件（如果存在）
                if hasattr(self, 'last_checkpoint') and os.path.exists(self.last_checkpoint):
                    os.remove(self.last_checkpoint)

                # 更新记录的上一次检查点文件名
                self.last_checkpoint = checkpoint_filename

                # 记录日志
                self.logger.info(f"Checkpoint saved for step {step}")
    
    def _train_step(self, iterator, model, optimizer, lr_scheduler, scaler, device, world_size):
        """
        执行单个训练步骤，包括前向传播、损失计算、反向传播、参数更新，以及指标的更新与计算。

        :param iterator: iterator, 用于获取当前批次数据的迭代器。
        :param model: torch.nn.Module, 当前模型。
        :param optimizer: torch.optim.Optimizer, 用于优化模型参数的优化器。
        :param lr_scheduler: torch.optim.lr_scheduler.LRScheduler, 学习率调度器，用于调整学习率。
        :param scaler: torch.cuda.amp.GradScaler, 用于混合精度训练时的梯度缩放。
        :param device: torch.device, 模型和数据的计算设备（如GPU）。
        :return: dict, 包含损失和计算出的指标的字典。
        """
        sample = self.task.load_sample(iterator, device)
        model.train()
        optimizer.zero_grad()

        # 使用自动混合精度（autocast）
        with torch.amp.autocast('cuda', enabled=self.fp16):
            result = self.task.train_step(model, sample)  # 执行训练步骤
            loss = result['loss']  # 提取损失

        # 如果启用了FP16，使用Scaler来缩放梯度
        if self.fp16:
            scaler.scale(loss).backward()  # 使用缩放后的梯度反向传播
            
            # 反缩放梯度
            scaler.unscale_(optimizer)  # 反缩放梯度
            
            # 更新参数
            scaler.step(optimizer)  # 更新参数
            scaler.update()  # 更新Scaler状态
        else:
            loss.backward()  # 不使用FP16，常规反向传播
            optimizer.step()  # 更新参数

        if lr_scheduler:
            lr_scheduler.step()

        # 如果是自监督模型，需要额外的步骤
        self.task.momentum_update(model)

        # 更新指标
        if 'output' in result:
            self._metrics_evaluator.update_metrics(result)  # 更新指标数据
            metrics = self._metrics_evaluator.calculate_metrics()  # 计算当前步骤的指标
        else:
            metrics = None

        # 聚合和平均损失与指标
        result = self._aggregate_metrics(loss, metrics, world_size, device)

        return result


    def _eval_step(self, model, loader, device, world_size):
        """
        执行单个评估步骤，包括模型评估、损失计算以及指标计算。

        :param model: torch.nn.Module, 当前模型。
        :param loader: DataLoader, 用于加载数据的迭代器。
        :param device: torch.device, 模型和数据的计算设备（如GPU）。
        :param world_size: int, 分布式训练中的总进程数，用于计算全局的平均损失。
        :return: dict, 包含总评估损失和计算出的指标。
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
                total += 1

                self._metrics_evaluator.update_metrics(result)  # 更新指标

            metrics = self._metrics_evaluator.calculate_metrics()  # 计算最终的指标

        # 聚合和平均损失与指标
        result = self._aggregate_metrics(tot_eval_loss, metrics, world_size, device, total)

        return result

    def _aggregate_metrics(self, loss, metrics, world_size, device, total=None):
        """
        聚合和平均损失与指标。

        :param loss: torch.Tensor 或 float, 当前步骤的损失。
        :param metrics: dict, 当前步骤的指标。
        :param world_size: int, 分布式训练中的总进程数。
        :param total: int, 样本总数（仅在评估步骤中使用）。
        :return: dict, 包含聚合和平均后的损失与指标的字典。
        """
        if world_size > 1:
            if not isinstance(loss, torch.Tensor):
                loss = torch.tensor(loss).to(device)
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss /= world_size

            if metrics is not None:
                for metric, value in metrics.items():
                    if isinstance(value, torch.Tensor):
                        dist.all_reduce(value, op=dist.ReduceOp.SUM)
                        value /= world_size
                        metrics[metric] = round(value.item(), 4)
                    elif isinstance(value, (int, float)):
                        tensor_value = torch.tensor(value).to(device)
                        dist.all_reduce(tensor_value, op=dist.ReduceOp.SUM)
                        metrics[metric] = round(tensor_value.item() / world_size, 4)
            else:
                metrics = {}

        if total is not None:
            if isinstance(loss, torch.Tensor):
                loss = loss.item()
            loss /= total

        # 确保 loss 是浮点数
        if isinstance(loss, torch.Tensor):
            loss = loss.item()

        result = {"loss": round(loss, 4)}
        result.update(metrics)

        return result


    def _state_dict(self, model, optimizer, lr_scheduler):
        return {
            "args": self.cfg,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict()
        }
    
    def _save_checkpoint(self, filename, model, optimizer, lr_scheduler):
        checkpoint = self._state_dict(model, optimizer, lr_scheduler)
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")

    def _load_checkpoint(self, filename):
        ckpt_params = torch.load(filename)
        self.args = ckpt_params["args"]
        self.model.load_state_dict(ckpt_params["model"])
        self.optimizer.load_state_dict(ckpt_params["optimizer"])
        self.lr_scheduler.load_state_dict(ckpt_params["lr_scheduler"])

    
        
