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
import torch
from utils import lazy_import_module, get_attr_from_cfg
from utils import build_dis_sampler, build_data_loader
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import os

class Trainer(object):
    def __init__(self, cfg) -> None:

        self.cfg = cfg

        # 分布式环境配置
        self.ddp_backend = get_attr_from_cfg(cfg,'trainer.ddp_backend','')
        

        # 自动混合精度配置
        self.fp16 = get_attr_from_cfg(cfg, 'trainer.fp16', False)

        # 训练配置
        self.total_epochs = get_attr_from_cfg(cfg, 'trainer.total_epochs', 0)
        self.save_epoch = get_attr_from_cfg(cfg, 'trainer.save_epoch', 0)
        self.valid_epoch = get_attr_from_cfg(cfg, 'trainer.valid_epoch', 0)
        self.batch_size = get_attr_from_cfg(cfg, 'trainer.batch_size', 0)

        # 任务配置
        self.task_select = get_attr_from_cfg(cfg, 'task.select', '')
        
        # 任务
        self.task = self.__build_task()
    
    def __build_task(self):

        task = lazy_import_module('tasks',self.task_select)

        return task(self.cfg)
    
    def __distributed_initializer(self, rank, world_size):
        # 设置主节点地址和端口
        os.environ['MASTER_ADDR'] = 'localhost'  # 或者设置为主节点的IP
        os.environ['MASTER_PORT'] = '12355'      # 选择一个合适的端口

        # 初始化分布式环境
        dist.init_process_group(backend=self.ddp_backend, rank=rank, world_size=world_size)

        # torch.cuda.set_device(rank)  # 设置当前进程所使用的 GPU

        


 
    def train(self,rank,world_size):

        # 初始化分布式进程
        self.distributed_initializer(rank=rank, world_size=world_size)

        # 加载数据集
        train_dataset = self.task.get_train_dataset()
        dev_dataset = self.task.get_dev_dataset()
        test_dataset = self.task.get_test_dataset()

        # 构造采样器
        train_sampler = build_dis_sampler(train_dataset, world_size, rank)
        # dev_sampler = build_dis_sampler(dev_dataset, world_size, rank)
        # test_sampler = build_dis_sampler(test_dataset, world_size, rank)

        # 构造数据加载器
        train_loader = build_data_loader(train_dataset, self.batch_size, train_sampler)
        dev_loader = build_data_loader(dev_dataset, self.batch_size,sampler=None)
        test_loader = build_data_loader(test_dataset, self.batch_size,sampler=None)

        # 加载优化器
        optimizer = self.task.get_optimizer()

        # 加载学习率调度器
        lr_scheduler = self.task.get_lr_scheduler()

        # 如果使用FP16，初始化 GradScaler
        scaler = GradScaler() if self.fp16 else None
        
        # 训练
        self.task.train()
        for epoch in range(self.total_epochs):
            self.__train_epoch(train_loader,optimizer,scaler,scaler,train_sampler,epoch)
            
            if rank == 0:
                lr_scheduler.step()

            if epoch % self.valid_epoch == 0:
                self.__eval_step(dev_loader,epoch)
            
            if epoch % self.save_epoch == 0:
                self.__save_checkpoint('state_dict')


        # 结束分布式进程
        dist.destroy_process_group()
    
    def __train_epoch(self, loader, optimizer, scaler, sampler, epoch):
        sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(loader):
                # data, target = data.to(rank), target.to(rank)
                optimizer.zero_grad()
                # 使用自动混合精度（autocast）
                with autocast(enabled=self.fp16):
                    result = self.task.train_step(data, target)
                    loss = result['loss']

                # 如果启用FP16，使用Scaler来缩放梯度
                if self.fp16:
                    scaler.scale(loss).backward()  # 使用缩放后的梯度反向传播
                    scaler.step(optimizer)  # 更新参数
                    scaler.update()  # 更新Scaler状态
                else:
                    loss.backward()  # 不使用FP16，常规反向传播
                    optimizer.step()  # 更新参数

                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {result['loss'].item()}")

    def __eval_step(self,loader, epoch):
        self.task.eval()
        total_val_loss = 0
        with torch.no_grad():  
            for batch_idx,(data, target) in enumerate(loader):  
                val_result = self.task.valid_step(data, target)
                val_loss = val_result['loss']
                total_val_loss += val_loss.item()
                
        avg_val_loss = total_val_loss / len(loader)  
        print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
        self.task.train()

    def __test_step(self,loader,epoch):
        pass

    def __save_checkpoint(self, filename):

        self.task.save_checkpoint(filename)


    def __load_checkpoint(self, filename):

        self.task.load_checkpoint(filename)

    
        
