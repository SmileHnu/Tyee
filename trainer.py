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
from utils import dynamic_import
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import os

class Trainer(object):
    def __init__(self, cfg) -> None:
        # 解析cfg获取train&distribution
        trainer_config = cfg.get('trainer',{})
        # self.world_size = trainer_config.get('world_size',1)
        self.ddp_backend = trainer_config.get('ddp_backend','')
        self.fp16 = trainer_config.get('fp16',False)
        
        self.total_epochs = trainer_config.get('total_epochs',0)
        self.valid_epoch = trainer_config.get('valid_epoch',0)
        self.save_epoch= trainer_config.get('save_epoch',0)

        # 解析cfg获取task
        task_config = cfg.get('task',{})
        self.task_select = task_config.get('select','')

        self.cfg = cfg
        # self.optimizer = self.get_optimizer(cfg)
        
        
        
    
    def load_task(self,cfg,rank,world_size):

        cls = dynamic_import(module_name='tasks',class_name=self.task_select)
        task = cls(cfg,rank,world_size)

        return task
    
    def distributed_initializer(self, rank, world_size):
        # 设置主节点地址和端口
        os.environ['MASTER_ADDR'] = 'localhost'  # 或者设置为主节点的IP
        os.environ['MASTER_PORT'] = '12355'      # 选择一个合适的端口

        # 初始化分布式环境
        dist.init_process_group(backend=self.ddp_backend, rank=rank, world_size=world_size)

        # torch.cuda.set_device(rank)  # 设置当前进程所使用的 GPU

        # 加载task
        self.task = self.load_task(cfg=self.cfg,rank=rank,world_size=world_size)


 
    def train(self,rank,world_size):

        # 初始化分布式进程
        self.distributed_initializer(rank=rank, world_size=world_size)

        # 加载dataloader
        train_loader, dev_loader, test_loader = self.task.load_data()
        # 加载optimizer
        optimizer = self.task.load_optimizer()
        # 加载scheduler
        lr_scheduler = self.task.load_lr_scheduler()
        # 加载sampler
        train_sampler, dev_sampler, test_sampler = self.task.load_sampler()

        # 如果使用FP16，初始化 GradScaler
        scaler = GradScaler() if self.fp16 else None
        
        # 训练
        self.task.train()
        for epoch in range(self.total_epochs):
            train_sampler.set_epoch(epoch)
            for batch_idx, (data, target) in enumerate(train_loader):
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
            
            if rank == 0:
                lr_scheduler.step()

            # if epoch % self.valid_epoch == 0:
            #     self.task.eval()
            #     total_val_loss = 0
            #     with torch.no_grad():  
            #         for batch_idx,(data, target) in enumerate(dev_loader):  
            #             val_loss = self.task.valid_step(data, target)
            #             total_val_loss += val_loss.item()
                
            #     avg_val_loss = total_val_loss / len(dev_loader)  
            #     print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
            #     self.task.train()
            # if epoch % self.save_epoch == 0:
            #     self.save_checkpoint('state_dict')


        # 结束分布式进程
        dist.destroy_process_group()
    
    

    def save_checkpoint(self, filename):

        checkpoint = self.task.state_dict()
        torch.save(checkpoint, filename)

        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):

        # checkpoint = torch.load(filename)
        # self.task.model.load_state_dict(checkpoint['model_state_dict'])

        # print(f"Checkpoint loaded from {filename}")
        pass

    
        
