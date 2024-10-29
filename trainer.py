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



class Trainer(object):
    def __init__(self, task, prop, batch_size) -> None:
        self.task = task
        self.train_loader, self.valid_loader = self.task.load_data(prop, batch_size)

    def train(self, total_epochs, valid_epoch, save_epoch):
        
        optimizer = self.task.get_optimizer()

        self.task.train()
        for epoch in range(total_epochs):
            
            for batch_idx, (data, target) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                loss = self.task.train_step(data, target)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch}, Loss: {loss.item()}")
            
            if epoch % valid_epoch == 0:
                self.task.eval()
                total_val_loss = 0
                with torch.no_grad():  
                    for batch_idx,(data, target) in enumerate(self.valid_loader):  
                        val_loss = self.task.valid_step(data, target)
                        total_val_loss += val_loss.item()
                
                avg_val_loss = total_val_loss / len(self.valid_loader)  
                print(f"Epoch {epoch}, Validation Loss: {avg_val_loss}")
                self.task.train()
            if epoch % save_epoch == 0:
                self.save_checkpoint('state_dict')
            
    
    def state_dict(self):

        return {
            # 优化器，超参数，
            'model_state_dict': self.task.model.state_dict(),
            'optimizer_state_dict': self.task.optimizer.state_dict()
        }

    def save_checkpoint(self, filename):

        checkpoint = self.state_dict()
        torch.save(checkpoint, filename)

        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):

        checkpoint = torch.load(filename)
        self.task.model.load_state_dict(checkpoint['model_state_dict'])

        print(f"Checkpoint loaded from {filename}")
    
