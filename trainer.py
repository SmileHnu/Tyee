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
import importlib


class Trainer(object):
    def __init__(self, cfg) -> None:
        self.task = self.load_task(cfg)
        # self.optimizer = self.get_optimizer(cfg)
        self.lr_scheduler = self.load_lr_scheduler(cfg)
        self.train_loader, self.dev_loader, self.test_loader = self.task.get_data()
        self.load_trainer(cfg)
    # 解析cfg获取train&distribution
    def load_trainer(self,cfg):
        trainer_config = cfg.get('trainer',{})
        self.ddp_backend = trainer_config.get('ddp_backend','')
        self.fp16 = trainer_config.get('fp16',False)
        self.world_size = trainer_config.get('world_size',1)
        self.total_epochs = trainer_config.get('total_epochs',0)
        self.valid_epoch = trainer_config.get('valid_epoch',0)
        self.save_epoch= trainer_config.get('save_epoch',0)


    # 解析cfg获取task
    def load_task(self,cfg):

        task_config = cfg.get('task',{})
        task_select = task_config.get('select','')

        cls = self.dynamic_import(module_name='tasks',class_name=task_select)
        task = cls(cfg)

        return task
    # 解析cfg获取lr_scheduler
    def load_lr_scheduler(self,cfg):
        
        lr_scheduler_config = cfg.get('lr_scheduler',{})
        lr_scheduler_select = lr_scheduler_config.get('select','')
        step_size = lr_scheduler_config.get('step_size', 20)  # 默认值
        gamma = lr_scheduler_config.get('gamma', 0.1)  # 默认值

        cls = self.dynamic_import(module_name='torch.optim.lr_scheduler', class_name=lr_scheduler_select)
        lr_scheduler = cls(self.task.get_optimizer(), step_size=step_size, gamma=gamma)

        return lr_scheduler
    def train(self,):
        
        optimizer = self.task.get_optimizer()

        self.task.train()
        for epoch in range(self.total_epochs):
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                loss = self.task.train_step(data, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
            
            self.lr_scheduler.step()

            # if epoch % self.valid_epoch == 0:
            #     self.task.eval()
            #     total_val_loss = 0
            #     with torch.no_grad():  
            #         for batch_idx,(data, target) in enumerate(self.dev_loader):  
            #             val_loss = self.task.valid_step(data, target)
            #             total_val_loss += val_loss.item()
                
            #     avg_val_loss = total_val_loss / len(self.dev_loader)  
            #     print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
            #     self.task.train()
            # if epoch % self.save_epoch == 0:
            #     self.save_checkpoint('state_dict')
            
    
    

    def save_checkpoint(self, filename):

        checkpoint = self.task.state_dict()
        torch.save(checkpoint, filename)

        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):

        # checkpoint = torch.load(filename)
        # self.task.model.load_state_dict(checkpoint['model_state_dict'])

        # print(f"Checkpoint loaded from {filename}")
        pass

    def dynamic_import(self, module_name, class_name):
        try:
            # 动态导入模块
            module = importlib.import_module(module_name)
            # 获取类，如果不存在则抛出 AttributeError
            cls = getattr(module, class_name)
            return cls
        except ImportError as e:
            raise ImportError(f"无法导入模块 '{module_name}': {e}")
        except AttributeError:
            raise AttributeError(f"模块 '{module_name}' 中没有找到类 '{class_name}'")
        
