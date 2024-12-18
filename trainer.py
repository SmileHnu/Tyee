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
from utils import init_logging
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from utils import lazy_import_module, get_nested_field, get_grad_norm
from torch.nn.parallel import DistributedDataParallel as DDP



class Trainer(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # 获取任务名称
        self.task_select = get_nested_field(cfg, 'task.select', '')
        
        # 获取实验保存路径
        root = get_nested_field(self.cfg, "common.exp_dir", "./experiments/")
        self.exp_dir = f"{root}/{datetime.datetime.now().strftime('%Y-%m-%d/%H-%M-%S')}"
        self.tb_dir, self.checkpoint_dir = self._create_experiment_directories()

        # 配置logging
        init_logging(self.exp_dir)
        self.logger = logging.getLogger(__name__)

        # TensorBoard日志
        self.writer = SummaryWriter(log_dir=self.tb_dir)

        # 保存config
        self.save_config()

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
        
        # 任务
        self.task = self._build_task()
        self._metrics = get_nested_field(cfg, 'trainer.metrics', ['accuracy_score'])


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
    
    def _create_experiment_directories(self):
        """创建实验目录结构，包括保存log, config, checkpoint, 和 TensorBoard文件夹。"""
        # 创建实验根目录
        os.makedirs(self.exp_dir, exist_ok=True)

        # 创建TensorBoard文件夹
        tb_dir = f"{self.exp_dir}/{self.task_select}_tb"
        os.makedirs(tb_dir, exist_ok=True)

        # 创建Checkpoint文件夹
        checkpoint_dir = f"{self.exp_dir}/checkpoint"
        os.makedirs(checkpoint_dir, exist_ok=True)

        return tb_dir, checkpoint_dir


    def save_config(self):
        """将配置文件保存到experiment目录下的config.yaml。"""
        import yaml
        config_file = os.path.join(self.exp_dir, "config.yaml")
        with open(config_file, "w", encoding='utf-8') as f:
            yaml.dump(self.cfg, f, default_flow_style=False, allow_unicode=True)
 
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
        if dev_dataset is not None:
            dev_loader = self.task.build_dataloader(dev_dataset, self._batch_size, sampler=None)
        else:
            dev_loader = None
        if test_dataset is not None:
            test_loader = self.task.build_dataloader(test_dataset, self._batch_size, sampler=None)
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
                sampler.set_epoch(step // len(train_loader))
                iterator = self.task.get_batch_iterator(train_loader)
             # 训练步骤
            train_result = self._train_step(iterator, model, optimizer, lr_scheduler, scaler, device)

            # 累积训练指标
            for metric, value in train_result.items():
                if metric not in accumulated_train_metrics:
                    accumulated_train_metrics[metric] = 0
                accumulated_train_metrics[metric] += value

            # 在更新参数之前，计算梯度范数
            grad_norm = get_grad_norm(model.parameters(), norm_type=2)  # 可以选择L2范数

            # 打印lr
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
            if step % self._log_interval == 0:
                # 计算平均训练指标
                avg_train_metrics = {
                    metric: round(value / self._log_interval, 4) for metric, value in accumulated_train_metrics.items()
                }
            
                # 打印训练日志
                self.logger.info(f"Train, Step {step}, Metrics: {avg_train_metrics}")
                # 重置累积值
                accumulated_train_metrics = {}

                # # 测试集日志
                # test_result = self._eval_step(model, test_loader, device, world_size)

                # # 打印测试日志
                # self.logger.info(f"Test, Step {step}, Metrics: {test_result}")

                # 写入 TensorBoard
                for metric, value in avg_train_metrics.items():
                    self.writer.add_scalar(f'train/{metric}', value, step)
                # # 输出模型的所有参数
                # for name, param in model.named_parameters():
                #     self.logger.info(f"Parameter Name: {name}, Parameter Shape: {param.shape}")
                #     self.logger.info(param.data)

            # 验证集日志
            if step % self._eval_interval == 0:
                
                # 验证集评估
                dev_result = self._eval_step(model, dev_loader, device, world_size)

                # 打印验证日志
                self.logger.info(f"Dev, Step {step}, Metrics: {dev_result}")

                # 写入 TensorBoard
                for metric, value in dev_result.items():
                    self.writer.add_scalar(f'dev/{metric}', value, step)

                # 测试集评估
                if test_loader is not None:
                    test_result = self._eval_step(model, test_loader, device, world_size)

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


            if step % self._save_interval == 0:
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



        # 结束分布式进程
        # dist.destroy_process_group()

    
    def _train_step(self, iterator, model, optimizer, lr_scheduler, scaler, device):
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


        # 更新指标
        self._metrics_evaluator.update_metrics(result)  # 更新指标数据
        metrics = self._metrics_evaluator.calculate_metrics()  # 计算当前步骤的指标

        # 将损失和指标合并为一个字典
        result = {"loss": loss.item()}
        result.update(metrics)

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

        # # 同步损失值，确保所有进程计算相同的损失
        # tot_eval_loss = torch.tensor(tot_eval_loss).to(device)
        # dist.all_reduce(tot_eval_loss, op=dist.ReduceOp.SUM)  # 聚合所有进程的损失
        # tot_eval_loss /= world_size  # 计算所有进程的平均损失

        # # 同步所有指标，确保所有进程计算相同的指标
        # for metric, value in metrics.items():
        #     if isinstance(value, torch.Tensor):  # 仅同步 tensor 类型的指标
        #         dist.all_reduce(value, op=dist.ReduceOp.SUM)  # 聚合所有进程的指标
        #         value /= world_size  # 计算所有进程的平均值

        # # 将所有指标转化为标量
        # metrics = {metric: value.item() for metric, value in metrics.items()}

        # 合并损失和指标到一个字典中
        result = {"loss": round(tot_eval_loss / total, 4)}
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

    
        
