from sklearn import model_selection
import numpy as np
from collections import Counter
import pandas as pd 

def train_test_split(
    info: pd.DataFrame, 
    test_size=None,
    train_size=None,
    random_state=None, 
    shuffle=True,
    stratify: str = None,
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

        if train_size is None and test_size is None:
            train_size = 0.75 
        elif test_size is not None and train_size is None:
            if not (0.0 < test_size < 1.0):
                raise ValueError("test_size must be a float between 0.0 and 1.0 if train_size is None")
            train_size = 1.0 - test_size
        elif train_size is not None and test_size is None:
            if not (0.0 < train_size < 1.0):
                raise ValueError("train_size must be a float between 0.0 and 1.0 if test_size is None")
        elif train_size is not None and test_size is not None:
            raise ValueError("Either test_size or train_size should be None, not both.")
        else: # Should not happen given the above logic
            raise ValueError("Invalid combination of train_size and test_size")


        train_indices = []
        test_indices = []

        for group_label, group_df in info.groupby(stratify, sort=False): 
            n_samples_in_group = len(group_df)
            n_train_for_group = int(round(n_samples_in_group * train_size))

            if n_train_for_group == 0 and n_samples_in_group > 0 and train_size > 0:
                n_train_for_group = 1
            
            n_test_for_group = n_samples_in_group - n_train_for_group
            if n_test_for_group == 0 and n_samples_in_group > 0 and (1.0 - train_size) > 0: # (1.0 - train_size) is test_size
                if n_train_for_group > 1 : 
                    n_train_for_group -=1


            group_original_indices = group_df.index.tolist()

            train_indices.extend(group_original_indices[:n_train_for_group])

            test_indices.extend(group_original_indices[n_train_for_group:])
        


        train_info_df = info.loc[train_indices].copy()
        test_info_df = info.loc[test_indices].copy()
        
    else: # shuffle=True or stratify is None or stratify column not in info
        if stratify and stratify not in info.columns:
            actual_stratify_data = None
            if shuffle: 
                print(f"Warning: Stratify column '{stratify}' not found in DataFrame. Proceeding without stratification.")
        elif stratify:
            actual_stratify_data = info[stratify]
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
            stratify=actual_stratify_data 
        )

    return train_info_df, test_info_df