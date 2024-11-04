import torch
from torch.utils.data import DataLoader, random_split
from . import PRLTask
from utils import dynamic_import
import os
from nn import UpstreamDownstreamModel

class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg):
        super().__init__()
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

        


        self.train_loader, self.dev_loader, self.test_loader = self.load_data()
        self.model = self.load_model()
        self.optimizer = self.load_optimizer()
        self.lr_scheduler = self.load_lr_scheduler()
        self.loss = self.load_loss()
        
    # 解析cfg构造dataloader
    def load_data(self, ):
        
        # 确定使用的dataset类
        Dataset = dynamic_import('dataset',self.dataset)

        # 确定训练集，测试集，验证集的路径
        train_path = os.path.join(self.path, self.train_path)
        dev_path = os.path.join(self.path,self.eval_path[0])
        test_path = os.path.join(self.path,self.eval_path[1])

        # 确定使用的transforms类
        transforms = []
        for t in self.transforms_select:
            t_transforms = dynamic_import('dataset.transforms',t)
            transforms.append(t_transforms)
        
        # 实例化dataset
        train_dataset = Dataset(train_path,transforms)
        dev_dataset = Dataset(dev_path,transforms)
        test_dataset = Dataset(test_path,transforms)

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=self.num_workers
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn, num_workers=self.num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=self.num_workers
        )
        return train_loader, dev_loader, test_loader

    
    def load_model(self,):
        

        # 实例化upstream
        upstream_cls = dynamic_import('models.upstream',self.upstream_select)
        upstream = upstream_cls()

        # 实例化downstream
        downstream_cls = dynamic_import('models.downstream',self.downstream_select)
        downstream = downstream_cls(output_dim = self.downstream_classes)

        model = UpstreamDownstreamModel(upstream=upstream,downstream=downstream,upstream_trainable=self.upstream_trainable)
        return model
        

    # 解析cfg构造loss
    def load_loss(self,):
        
        cls = dynamic_import('torch.nn',self.loss_select)

        if self.loss_weight:
            return cls(weight=torch.tensor(self.loss_weight, dtype=torch.float32))
        else:
            return cls()

    
    def load_optimizer(self,):
        

        cls = dynamic_import('torch.optim',self.optimizer_select)

        optimizer = cls(self.model.parameters(),lr=self.lr)

        return optimizer
    

    def load_lr_scheduler(self,):
        

        cls = dynamic_import(module_name='torch.optim.lr_scheduler', class_name=self.lr_scheduler_select)
        lr_scheduler = cls(self.optimizer, step_size=self.step_size, gamma=self.gamma)

        return lr_scheduler

    def get_data(self, ):
        return self.train_loader, self.dev_loader, self.test_loader
        
    def get_optimizer(self, ):
        return self.optimizer
    
    def get_lr_scheduler(self,):
        return self.lr_scheduler

    def train_step(self, data, target):
        # print(type(data),type(target))
        output,_ = self.model(data, data.size(1))
        loss = self.loss(output,target)

        return loss
        

    def valid_step(self, data, target):
        # print(type(data),type(target))
        output, _ = self.model(data)
        loss = self.loss(output,target)

        return loss
    
    
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
    

        