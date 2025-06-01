from sklearn import model_selection
import numpy as np
from collections import Counter
import torch # 如果你后续操作需要torch，可以保留，否则可以移除
import pandas as pd # 需要导入 pandas

def train_test_split(
    info: pd.DataFrame, # 明确 info 是 DataFrame
    test_size=None,
    train_size=None,
    random_state=None, # 在自定义逻辑中未使用，仅用于 sklearn 回退
    shuffle=True,
    stratify: str = None, # stratify 现在是 info DataFrame 中的列名
): 
    """Splits data into training and testing sets. 

    If shuffle=False and stratify (column name) is not None, 
    the function will split the data such that each class (from the stratify column in info) 
    is represented in the training set according to train_size, taking samples in their original order.
     
    If shuffle=True or stratify is None, the function will default to using 
    the sklearn.model_selection.train_test_split function.

    Args:
        info (pd.DataFrame): The DataFrame containing all samples.
        test_size (float, optional): Proportion to take for the test set. Defaults to None.
        train_size (float, optional): Proportion to take for the train set. Defaults to None.
        random_state (int, optional): Random state for sklearn's split. Defaults to None.
        shuffle (bool, optional): Whether to shuffle data (triggers sklearn's split if True). Defaults to True.
        stratify (str, optional): The column name in 'info' to use for stratification. Defaults to None.

    Returns:
        tuple: (train_info_df, test_info_df)
    """

    if not shuffle and stratify and stratify in info.columns:
        # 自定义不打乱顺序的分层采样逻辑

        if train_size is None and test_size is None:
            train_size = 0.75 # 默认训练集比例
        elif test_size is not None and train_size is None:
            if not (0.0 < test_size < 1.0):
                raise ValueError("test_size must be a float between 0.0 and 1.0 if train_size is None")
            train_size = 1.0 - test_size
        elif train_size is not None and test_size is None:
            if not (0.0 < train_size < 1.0):
                raise ValueError("train_size must be a float between 0.0 and 1.0 if test_size is None")
            # test_size 将是 1.0 - train_size
        elif train_size is not None and test_size is not None:
            raise ValueError("Either test_size or train_size should be None, not both.")
        else: # Should not happen given the above logic
            raise ValueError("Invalid combination of train_size and test_size")


        train_indices = []
        test_indices = []

        # 获取分层依据列中的唯一类别及其在原始 DataFrame 中的索引
        for group_label, group_df in info.groupby(stratify, sort=False): # sort=False 尽量保持原始组顺序
            n_samples_in_group = len(group_df)
            n_train_for_group = int(round(n_samples_in_group * train_size))
            
            # 如果计算出的训练样本数为0，但组内有样本，至少分配1个给训练集（如果训练集大小允许）
            # 或者如果测试样本数为0，但组内有样本，至少分配1个给测试集
            # 这有助于避免小类别完全丢失，但需小心处理 train_size 极小或极大的情况
            if n_train_for_group == 0 and n_samples_in_group > 0 and train_size > 0:
                n_train_for_group = 1
            
            n_test_for_group = n_samples_in_group - n_train_for_group
            if n_test_for_group == 0 and n_samples_in_group > 0 and (1.0 - train_size) > 0: # (1.0 - train_size) is test_size
                if n_train_for_group > 1 : # 确保训练集至少有1个后，才调整
                    n_train_for_group -=1
                # n_test_for_group 会变成 1

            # 获取当前组的原始索引（这些索引对应于原始 info DataFrame）
            group_original_indices = group_df.index.tolist()

            # 按原始顺序取训练样本的索引
            train_indices.extend(group_original_indices[:n_train_for_group])
            # 按原始顺序取测试样本的索引
            test_indices.extend(group_original_indices[n_train_for_group:])
        
        # 使用收集到的索引从原始 info DataFrame 中提取训练集和测试集
        # 先对索引排序，以确保最终的 DataFrame 保持原始 info 中样本的相对顺序（如果需要的话）
        # 如果严格按类别顺序拼接，则不需要排序 train_indices 和 test_indices
        # train_indices.sort() # 可选，如果希望最终输出的行顺序与原始info一致
        # test_indices.sort()  # 可选

        train_info_df = info.loc[train_indices].copy()
        test_info_df = info.loc[test_indices].copy()
        
    else: # shuffle=True or stratify is None or stratify column not in info
        if stratify and stratify not in info.columns:
            # 如果提供了 stratify 列名但该列不存在，则不能进行分层，回退到非分层划分
            actual_stratify_data = None
            if shuffle: # 只有在 shuffle 为 True 时才警告，因为 False 时自定义逻辑不会执行
                print(f"Warning: Stratify column '{stratify}' not found in DataFrame. Proceeding without stratification.")
        elif stratify:
            actual_stratify_data = info[stratify]
            # 检查 stratify 列是否有 NaN，sklearn 不允许
            if actual_stratify_data.isnull().any():
                print(f"Warning: Stratify column '{stratify}' contains NaN values. Sklearn's train_test_split might fail or behave unexpectedly. Consider cleaning NaNs.")
        else:
            actual_stratify_data = None
        
        train_info_df, test_info_df = model_selection.train_test_split(
            info,
            test_size=test_size,
            train_size=train_size,
            random_state=random_state,
            shuffle=shuffle,
            stratify=actual_stratify_data # 使用实际的 Series 数据进行分层
        )

    return train_info_df, test_info_df