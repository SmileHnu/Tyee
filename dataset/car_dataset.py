import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from dataset.base_dataset import BaseDataset
class CarDataset(BaseDataset):
    def __init__(self):
        self.features, self.target = self.get_data() 

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        input = self.features[idx]
        target = self.target[idx]
        return input, target
    
    def get_data(self):
        data_df = pd.read_csv('data\car_1000.txt',delimiter=',',names=['buying','maint','doors','persons','lug_boot','safety','target'])

        # 标签编码
        label_encoder = LabelEncoder()
        encoded_df = data_df.apply(label_encoder.fit_transform)
        # print(encoded_df)
        features = encoded_df.iloc[:,:-1].values
        targets = encoded_df.iloc[:,-1:].values
        # print(targets.shape)
        targets =targets.squeeze(1)

        return features,targets
    
    def collate_fn(self,batch):
        data, targets = zip(*batch)  # 解压数据和目标
        data = [torch.tensor(d, dtype=torch.float32) for d in data]  # 将每个 NumPy 数组转换为 PyTorch 张量
        targets = torch.tensor(targets,dtype=torch.long)  # 目标也转换为张量
        return torch.stack(data), targets  # 返回堆叠后的数据和目标
        

if __name__ == "__main__":
    
    dataset = CarDataset()
    data, target = dataset[0]
    print(data)
    pass