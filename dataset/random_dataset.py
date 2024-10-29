import torch
from .base_dataset import BaseDataset

class RandomDataset(BaseDataset):
    def __init__(self, num_samples=1000, input_dim=10, num_classes=2):
        self.num_samples = num_samples
        self.data = torch.randn(num_samples, input_dim)  
        self.targets = torch.randint(0, num_classes, (num_samples,))  

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # print(type(self.data[idx]))
        return self.data[idx], self.targets[idx]
    
    def collate_fn(self,batch):
        data, targets = zip(*batch)  # 解压数据和目标
        data = torch.stack(data)  # 将数据转换为张量
        targets = torch.tensor(targets)  # 将目标转换为张量
        return data, targets


# if __name__ == "__main__":
    
#     dataset = RandomDataset()
#     data, target = dataset[0]
#     print(type(data))
#     pass