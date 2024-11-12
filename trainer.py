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
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import lazy_import_module, get_attr_from_cfg, build_dis_sampler, build_data_loader


class Trainer(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # 分布式环境配置
        self._ddp_backend = get_attr_from_cfg(self.cfg, "trainer.ddp_backend", "nccl")
        self._init_method = get_attr_from_cfg(self.cfg, "trainer.init_method", "tcp://127.0.0.1:60575")

        # 自动混合精度配置
        self.fp16 = get_attr_from_cfg(cfg, 'trainer.fp16', False)

        # 训练配置
        self._total_steps = get_attr_from_cfg(cfg, 'trainer.total_steps', 100)
        self._save_interval = get_attr_from_cfg(cfg, 'trainer.save_interval', 10)
        self._eval_interval = get_attr_from_cfg(cfg, 'trainer.eval_interval', 10)
        self._log_interval = get_attr_from_cfg(cfg, 'trainer.log_interval', 10)
        self._batch_size = get_attr_from_cfg(cfg, 'dataset.batch_size', 1)

        # 任务配置
        self.task_select = get_attr_from_cfg(cfg, 'task.select', '')
        
        # 任务
        self.task = self._build_task()

        # 加载数据集
        self.train_dataset = self.task.get_train_dataset()
        self.dev_dataset = self.task.get_dev_dataset()
        self.test_dataset = self.task.get_test_dataset()

        # 模型、优化器与学习率调度器
        self.model = self.task.build_model()
        self.optimizer = self.task.build_optimizer(self.model)
        self.lr_scheduler = self.task.build_lr_scheduler(self.optimizer)

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

        # 初始化分布式进程
        # self.distributed_initializer(rank=rank, world_size=world_size)
        device = torch.device(rank)
        self.model = self.model.to(device)

        # 构造采样器
        sampler = build_dis_sampler(self.train_dataset, world_size, rank)

        # 构造数据加载器
        train_loader = build_data_loader(self.train_dataset, self._batch_size, sampler=sampler)
        dev_loader = build_data_loader(self.dev_dataset, self._batch_size, sampler=None)
        test_loader = build_data_loader(self.test_dataset, self._batch_size, sampler=None)


        # 如果使用FP16，初始化 GradScaler
        scaler = GradScaler() if self.fp16 else None

        iterator = iter(train_loader)
        for step in range(self._total_steps):
            if step % len(train_loader) == 0:
                sampler.set_epoch(step // len(train_loader))
            
            try:
                sample = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                sample = next(iterator)

            for k, v in sample.items():
                if v is not None:
                    sample[k] = v.to(device)
    
            self.model.train()
            self.optimizer.zero_grad()
            # 使用自动混合精度（autocast）
            tot_loss = 0
            with autocast(enabled=self.fp16):
                result = self.task.train_step(self.model, sample)
                loss = result['loss']
                tot_loss += loss.item()

            # 如果启用FP16，使用Scaler来缩放梯度
            if self.fp16:
                scaler.scale(loss).backward()  # 使用缩放后的梯度反向传播
                scaler.step(self.optimizer)  # 更新参数
                scaler.update()  # 更新Scaler状态
            else:
                loss.backward()  # 不使用FP16，常规反向传播
                self.optimizer.step()  # 更新参数

            if step % self._log_interval == 0:
                print(f"Step {step}, Loss: {tot_loss / self._log_interval}")
                tot_loss = 0
                y_true, y_pred, y_scores = [], [], []

                self.model.eval()
                with torch.no_grad():
                    total, correct = 0, 0
                    tot_eval_loss = 0
                    for sample in iter(test_loader):
                        for k, v in sample.items():
                            if v is not None:
                                sample[k] = v.to(device)
                        result = self.task.valid_step(self.model, sample)
                        tot_eval_loss += result['loss'].item()
                        total += len(result["target"])

                        probabilities = result["pred"].softmax(dim=-1)  # 获取每个类别的概率
                        # 保存结果
                        y_true.extend(result["target"].cpu())
                        y_pred.extend(result["pred"].cpu().argmax(dim=-1))  # 将预测结果转为标签
                        y_scores.extend(probabilities[:, 1].cpu())  # 假设我们对二分类任务，只关注正类概率
                # acc = (correct / total).item()
                # print(f"Step {step}, Test Loss: {tot_eval_loss / total}, acc: {acc}")
                print(f"Step {step}, Test Loss: {tot_eval_loss / total}, acc: {accuracy_score(y_true, y_pred)}, auroc: {roc_auc_score(y_true, y_scores)}")
            pass

        # 训练
        # self.model.train()
        # for epoch in range(self.total_steps):
        #     self._train_epoch(train_loader, scaler, train_sampler, epoch, rank)
            
        #     self.lr_scheduler.step()

        #     if epoch % self.eval_step == 0:
        #         self.eval_step(dev_loader, epoch, rank)
            
        #     if epoch % self.eval_step == 0:
        #         self.eval_step(test_loader, epoch, rank)
            
        #     if epoch % self.save_step == 0:
        #         self._save_checkpoint('state_dict')


        # 结束分布式进程
        # dist.destroy_process_group()
    
    def _train_epoch(self, itr, scaler, epoch, rank):
        self.model.train()
        for batch_idx, samples in enumerate(loader):
            self.optimizer.zero_grad()
            # 使用自动混合精度（autocast）
            with autocast(enabled=self.fp16):
                result = self.task.train_step(self.model, samples)
                loss = result['loss']

            # 如果启用FP16，使用Scaler来缩放梯度
            if self.fp16:
                scaler.scale(loss).backward()  # 使用缩放后的梯度反向传播
                scaler.step(self.optimizer)  # 更新参数
                scaler.update()  # 更新Scaler状态
            else:
                loss.backward()  # 不使用FP16，常规反向传播
                self.optimizer.step()  # 更新参数

            if batch_idx % 100 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Loss: {result['loss'].item()}")

    def _eval_step(self,loader, epoch, rank):
        tot_val_loss = 0
        self.model.eval()
        with torch.no_grad():  
            for idx, samples in enumerate(loader):  
                outs = self.task.valid_step(self.model, samples)
                loss = outs['loss']
                tot_val_loss += loss.item()
                
        avg_val_loss = tot_val_loss / len(loader)  
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")

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

    
        
