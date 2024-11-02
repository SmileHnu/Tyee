import torch
from torch.utils.data import DataLoader, random_split
from . import PRLTask
import importlib
import os
class EmotionRecognitionTask(PRLTask):
    def __init__(self, cfg):
        '''
        dataset:
        model:
        optimizer:
        loss:
        '''
        super().__init__()
        self.train_loader, self.dev_loader, self.test_loader = self.load_data(cfg)
        # self.upstream, self.downstream = self.load_model(cfg)
        self.downstream = self.load_model(cfg)
        self.optimizer = self.load_optimizer(cfg)
        self.loss = self.load_loss(cfg)
        
    # 解析cfg构造dataloader
    def load_data(self, cfg ):
        dataset_config = cfg.get('dataset',{})
        num_workers = dataset_config.get('num_workers',1)
        path = dataset_config.get('path','')
        train = dataset_config.get('train','')
        eval = dataset_config.get('eval','')
        transforms_config = dataset_config.get('transforms','')
        transforms_select = transforms_config.get('select','')
        dataset = dataset_config.get('dataset','')
        batch_size = dataset_config.get('batch_size',1)

        # 确定使用的dataset类
        Dataset = self.dynamic_import('dataset',dataset)

        # 确定训练集，测试集，验证集的路径
        train_path = os.path.join(path, train)
        dev_path = os.path.join(path,eval[0])
        test_path = os.path.join(path,eval[1])

        # 确定使用的transforms类
        transforms = []
        for t in transforms_select:
            t_transforms = self.dynamic_import('dataset.transforms',t)
            transforms.append(t_transforms)
        
        # 实例化dataset
        train_dataset = Dataset(train_path,transforms)
        dev_dataset = Dataset(dev_path,transforms)
        test_dataset = Dataset(test_path,transforms)

        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn, num_workers=num_workers
        )
        dev_loader = DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=False, collate_fn=dev_dataset.collate_fn, num_workers=num_workers
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_fn, num_workers=num_workers
        )
        return train_loader, dev_loader, test_loader

    # 解析cfg构造model
    def load_model(self,cfg):
        model_config = cfg.get('model',{})
        downstream_config = model_config.get('downstream',{})
        downstream_classes = downstream_config.get('classes',1)
        downstream_select = downstream_config.get('select','')
        upstream_config = model_config.get('upstream',{})
        upstream_select = upstream_config.get('select','')

        # 实例化upstream
        # upstream_cls = self.dynamic_import('models.upstream',upstream_select)
        # upstream = upstream_cls()

        # 实例化downstream
        downstream_cls = self.dynamic_import('models.downstream',downstream_select)
        downstream = downstream_cls(output_dim = downstream_classes)

        # return upstream, downstream
        return downstream

    # 解析cfg构造loss
    def load_loss(self,cfg):
        task_config = cfg.get('task',{})
        loss_config = task_config.get('loss',{})
        loss_select = loss_config.get('select','')
        loss_weight = loss_config.get('weight',[])

        cls = self.dynamic_import('torch.nn',loss_select)

        if loss_weight:
            return cls(weight=torch.tensor(loss_weight, dtype=torch.float32))
        else:
            return cls()

    # 解析cfg构造optimizer
    def load_optimizer(self,cfg):
        optimizer_config = cfg.get('optimizer',{})
        optimizer_select = optimizer_config.get('select','')
        lr = optimizer_config.get('lr',0.01)

        cls = self.dynamic_import('torch.optim',optimizer_select)

        optimizer = cls(self.downstream.parameters(),lr=lr)

        return optimizer
    

    def get_data(self, ):
        return self.train_loader, self.dev_loader, self.test_loader
        
    def get_optimizer(self, ):
        return self.optimizer

    def train_step(self, data, target):
        # print(type(data),type(target))
        output = self.downstream(data)
        loss = self.loss(output,target)

        return loss
        

    def valid_step(self, data, target):
        # print(type(data),type(target))
        output = self.downstream(data)
        loss = self.loss(output,target)

        return loss
    
    
    def train(self,) -> None:
        self.downstream.train()
    
    def eval(self,) -> None:
        self.downstream.eval()
    
    def state_dict(self):

        return {
            # 优化器，超参数，
            'downstream_state_dict': self.downstream.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
    
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
        