import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from . import BaseDataset
from sklearn.model_selection import train_test_split

class CarDataset(BaseDataset):
    def __init__(self, path, transforms=None, split="train", train_ratio=0.8, dev_ratio=0.1, random_state=42):
        """
        初始化数据集并随机划分为 train/dev/test。

        Args:
            path (str): 数据文件路径。
            transforms (callable, optional): 数据增强或预处理操作。
            split (str): 数据集划分 ("train", "dev", "test")。
            train_ratio (float): 训练集比例。
            dev_ratio (float): 验证集比例。
            random_state (int): 随机种子，确保划分可复现。
        """
        self.path = path
        self.transforms = transforms
        self.split = split

        # 获取完整数据
        features, targets = self.get_data()

        # 首先划分 train 和 temp (dev + test)
        train_features, temp_features, train_targets, temp_targets = train_test_split(
            features, targets, test_size=(1 - train_ratio), random_state=random_state
        )

        # 再划分 temp 为 dev 和 test
        dev_size = dev_ratio / (1 - train_ratio)  # dev 比例相对 temp 数据集
        dev_features, test_features, dev_targets, test_targets = train_test_split(
            temp_features, temp_targets, test_size=(1 - dev_size), random_state=random_state
        )

        # 根据 split 参数选择对应的数据集
        if split == "train":
            self.features, self.target = train_features, train_targets
        elif split == "dev":
            self.features, self.target = dev_features, dev_targets
        elif split == "test":
            self.features, self.target = test_features, test_targets
        else:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'dev', or 'test'.")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        input = self.features[idx]
        target = self.target[idx]
        return input, target

    def get_data(self):
        data_df = pd.read_csv(self.path, delimiter=',', names=['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'target'])

        # 标签编码
        label_encoder = LabelEncoder()
        encoded_df = data_df.apply(label_encoder.fit_transform)
        features = encoded_df.iloc[:, :-1].values
        targets = encoded_df.iloc[:, -1].values

        return features, targets
    
    def collate_fn(self,batch):
        data, targets = zip(*batch)  # 解压数据和目标
        data = [torch.tensor(d, dtype=torch.float32) for d in data]  # 将每个 NumPy 数组转换为 PyTorch 张量
        targets = torch.tensor(targets,dtype=torch.long)  # 目标也转换为张量
       # 返回字典
        return {
            'x': torch.stack(data),  # 堆叠输入数据
            'target': targets          # 目标张量
        }
        

if __name__ == "__main__":
    
    dataset = CarDataset()
    data, target = dataset[0]
    print(data)
    pass