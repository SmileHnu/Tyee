#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : task.py
@Time    : 2024/09/25 16:41:26
@Desc    : 
"""
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import dynamic_import
import os
from nn import UpstreamDownstreamModel

class PRLTask(object):
    def __init__(self, cfg, rank, world_size) -> None:
        # 解析cfg构造dataloader
        dataset_config = cfg.get('dataset',{})
        self.num_workers = dataset_config.get('num_workers',1)
        self.path = dataset_config.get('path','')
        self.train_path = dataset_config.get('train','')
        self.eval_path = dataset_config.get('eval','')
        transforms_config = dataset_config.get('transforms','')
        self.transforms_select = transforms_config.get('select','')
        self.dataset = dataset_config.get('dataset','')
        self.batch_size = dataset_config.get('batch_size',1)

        # 解析cfg构造model
        model_config = cfg.get('model',{})
        downstream_config = model_config.get('downstream',{})
        self.downstream_classes = downstream_config.get('classes',1)
        self.downstream_select = downstream_config.get('select','')
        upstream_config = model_config.get('upstream',{})
        self.upstream_select = upstream_config.get('select','')
        self.upstream_trainable = upstream_config.get('trainable',False)

        # 解析cfg构造loss
        task_config = cfg.get('task',{})
        loss_config = task_config.get('loss',{})
        self.loss_select = loss_config.get('select','')
        self.loss_weight = loss_config.get('weight',[])

        # 解析cfg构造optimizer
        optimizer_config = cfg.get('optimizer',{})
        self.optimizer_select = optimizer_config.get('select','')
        self.lr = optimizer_config.get('lr',0.01)

        # 解析cfg构造lr_scheduler
        lr_scheduler_config = cfg.get('lr_scheduler',{})
        self.lr_scheduler_select = lr_scheduler_config.get('select','')
        self.step_size = lr_scheduler_config.get('step_size', 20)  # 默认值
        self.gamma = lr_scheduler_config.get('gamma', 0.1)  # 默认值

        # 分布式环境参数
        self.rank = rank
        self.world_size = world_size
        # 使用 DistributedSampler 以确保数据被均匀分布到各个进程
        train_dataset = self.build_dataset(os.path.join(self.path, self.train_path),self.transforms_select)
        dev_dataset = self.build_dataset(os.path.join(self.path,self.eval_path[0]),self.transforms_select)
        test_dataset = self.build_dataset(os.path.join(self.path,self.eval_path[1]),self.transforms_select)

        self.train_sampler = self.build_sampler(train_dataset)
        self.dev_sampler = self.build_sampler(dev_dataset)
        self.test_sampler = self.build_sampler(test_dataset)

        self.train_loader = self.build_dataloader(train_dataset,self.train_sampler)
        self.dev_loader = self.build_dataloader(dev_dataset,self.dev_sampler)
        self.test_loader = self.build_dataloader(test_dataset,self.test_sampler)

        self.model = self.build_model()
        self.optimizer = self.build_optimizer()
        self.lr_scheduler = self.build_lr_scheduler()
        self.loss = self.build_loss()
    

    def build_dataset(self,path,transforms_select):
        # 确定使用的dataset类
        Dataset = dynamic_import('dataset',self.dataset)

        # 确定使用的transforms类
        transforms = []
        for t in transforms_select:
            t_transforms = dynamic_import('dataset.transforms',t)
            transforms.append(t_transforms)
        
        dataset = Dataset(path,transforms)

        return dataset
    
    def build_sampler(self,dataset):
        sampler = DistributedSampler(dataset, num_replicas=self.world_size, rank=self.rank)

        return sampler
    def build_dataloader(self,dataset,sampler):
        loader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            collate_fn=dataset.collate_fn, 
            num_workers=self.num_workers,
            sampler=sampler
        )
        return loader
    
    def build_model(self,):
        
        # 实例化upstream
        upstream_cls = dynamic_import('models.upstream',self.upstream_select)
        upstream = upstream_cls()

        # 实例化downstream
        downstream_cls = dynamic_import('models.downstream',self.downstream_select)
        downstream = downstream_cls(output_dim = self.downstream_classes)
        
        model = UpstreamDownstreamModel(upstream=upstream,downstream=downstream,upstream_trainable=self.upstream_trainable)
        model = model.to(self.rank)
        ddp_model = DDP(model,device_ids=[self.rank])
        return ddp_model
    
    # 解析cfg构造loss
    def build_loss(self,):
        
        cls = dynamic_import('torch.nn',self.loss_select)

        if self.loss_weight:
            return cls(weight=torch.tensor(self.loss_weight, dtype=torch.float32).to(self.rank))
        else:
            return cls()

    def build_optimizer(self,):
        
        cls = dynamic_import('torch.optim',self.optimizer_select)

        optimizer = cls(self.model.parameters(),lr=self.lr)

        return optimizer
    

    def build_lr_scheduler(self,):
        
        cls = dynamic_import(module_name='torch.optim.lr_scheduler', class_name=self.lr_scheduler_select)
        lr_scheduler = cls(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        return lr_scheduler


    def load_data(self, ):
      
        return self.train_loader, self.dev_loader, self.test_loader
        
    def load_optimizer(self, ):
        return self.optimizer
    
    def load_lr_scheduler(self,):
        return self.lr_scheduler
    
    def load_sampler(self,):
        return self.train_sampler, self.dev_sampler, self.test_sampler
    def train(self,) -> None:
        self.model.train()
    
    def eval(self,) -> None:
        self.model.eval()

    def state_dict(self):

        return {
            # 优化器，超参数，
            'downstream_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
    def save_checkpoint(self, filename):

        checkpoint = self.task.state_dict()
        torch.save(checkpoint, filename)

        print(f"Checkpoint saved to {filename}")

    def load_checkpoint(self, filename):

        # checkpoint = torch.load(filename)
        # self.task.model.load_state_dict(checkpoint['model_state_dict'])

        # print(f"Checkpoint loaded from {filename}")
        pass
    def train_step(self, ):
        pass

    
    def valid_step(self, ):
        pass   
    
