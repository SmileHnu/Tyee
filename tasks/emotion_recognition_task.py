import torch
from torch.utils.data import DataLoader, random_split
from tasks.task import PRLTask

class EmotionRecognitionTask(PRLTask):
    def __init__(self, dataset, model, optimizer, loss):
        '''
        dataset:
        model:
        optimizer:
        loss:
        '''
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.optimizer = optimizer
        self.loss = loss
        
        
    def load_data(self, prop, batch_size ):

        # 将数据集分为训练集和验证集
        train_size = int(prop * len(self.dataset))
        valid_size = len(self.dataset) - train_size
        train_dataset, valid_dataset = random_split(self.dataset, [train_size, valid_size])
        # 创建 DataLoader
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, collate_fn=self.dataset.collate_fn
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=self.dataset.collate_fn
        )
        
        return train_loader, valid_loader
        

    def train_step(self, data, target):
        # print(type(data),type(target))
        output = self.model(data)
        loss = self.loss(output,target)

        return loss
        

    def valid_step(self, data, target):
        # print(type(data),type(target))
        output = self.model(data)
        loss = self.loss(output,target)

        return loss
    

    def get_optimizer(self, ):
        return self.optimizer
    
    def train(self,) -> None:
        self.model.train()
    
    def eval(self,) -> None:
        self.model.eval()