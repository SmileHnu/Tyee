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


class Trainer(object):
    def __init__(self, cfg) -> None:
        # 解析cfg获取train&distribution
        trainer_config = cfg.get('trainer',{})
        self.ddp_backend = trainer_config.get('ddp_backend','')
        self.fp16 = trainer_config.get('fp16',False)
        self.world_size = trainer_config.get('world_size',1)
        self.total_epochs = trainer_config.get('total_epochs',0)
        self.valid_epoch = trainer_config.get('valid_epoch',0)
        self.save_epoch= trainer_config.get('save_epoch',0)

        # 解析cfg获取task
        task_config = cfg.get('task',{})
        self.task_select = task_config.get('select','')

        self.task = self.load_task(cfg)
        # self.optimizer = self.get_optimizer(cfg)
        
        self.train_loader, self.dev_loader, self.test_loader = self.task.get_data()
        
    
    def load_trainer(self,):
        pass


    
    def load_task(self,cfg):

        cls = dynamic_import(module_name='tasks',class_name=self.task_select)
        task = cls(cfg)

        return task
    
 
    def train(self,):
        
        optimizer = self.task.get_optimizer()
        lr_scheduler = self.task.get_lr_scheduler()
        self.task.train()
        for epoch in range(self.total_epochs):
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                loss = self.task.train_step(data, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
            
            lr_scheduler.step()

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

    
        
