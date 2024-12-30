import os
import numpy as np
import pandas as pd
from glob import glob
import pickle
from scipy.signal import resample

def load_data_from_files(eeg_file, joints_file):
    eeg_data = pd.read_csv(eeg_file, delimiter="\t", header=None, skiprows=1)
    eeg_data = eeg_data.iloc[:, :-5]  # 去掉无用的列

    joints_data = pd.read_csv(joints_file, delimiter="\t", header=None, skiprows=2)
    joints_data = joints_data.iloc[:, :-1]  # 去掉时间戳列

    timestamps = eeg_data.iloc[:, 0].values

    # 划分数据集
    train_start_time = 2 * 60
    train_end_time = 12 * 60
    val_start_time = 12 * 60
    val_end_time = 17 * 60
    test_start_time = 17 * 60
    test_end_time = 22 * 60

    train_idx = (timestamps >= train_start_time) & (timestamps < train_end_time)
    val_idx = (timestamps >= val_start_time) & (timestamps < val_end_time)
    test_idx = (timestamps >= test_start_time) & (timestamps <= test_end_time)

    train_eeg_data = eeg_data[train_idx]
    val_eeg_data = eeg_data[val_idx]
    test_eeg_data = eeg_data[test_idx]

    train_joints_data = joints_data[train_idx]
    val_joints_data = joints_data[val_idx]
    test_joints_data = joints_data[test_idx]

    train_joints_data = train_joints_data.iloc[:, 1:]
    val_joints_data = val_joints_data.iloc[:, 1:]
    test_joints_data = test_joints_data.iloc[:, 1:]

    train_eeg_data = train_eeg_data.iloc[:, 1:]
    val_eeg_data = val_eeg_data.iloc[:, 1:]
    test_eeg_data = test_eeg_data.iloc[:, 1:]

    # 对EEG数据进行重采样
    train_eeg_data_resampled = resample_data(train_eeg_data)
    val_eeg_data_resampled = resample_data(val_eeg_data)
    test_eeg_data_resampled = resample_data(test_eeg_data)

    train_joints_data_resampled = resample_data(train_joints_data)
    val_joints_data_resampled = resample_data(val_joints_data)
    test_joints_data_resampled = resample_data(test_joints_data)

    train_joints_data_resampled = normalize_angles(train_joints_data_resampled)
    val_joints_data_resampled = normalize_angles(val_joints_data_resampled)
    test_joints_data_resampled = normalize_angles(test_joints_data_resampled)

    return (train_eeg_data_resampled, train_joints_data_resampled), \
           (val_eeg_data_resampled, val_joints_data_resampled), \
           (test_eeg_data_resampled, test_joints_data_resampled)

def resample_data(data, target_fs=200, fs=100):
    num_samples = int(data.shape[0] * target_fs / fs)
    resampled_data = np.zeros((num_samples, data.shape[1]))

    for i in range(data.shape[1]):
        resampled_data[:, i] = resample(data.iloc[:, i], num_samples)

    return pd.DataFrame(resampled_data, columns=data.columns)

def normalize_angles(joints_data):
    return joints_data / 90.0

def save_data(eeg_data, joints_data, split, dump_folder, subject_id, trial_id, stride=10, window_size=400):
    trials = []
    joint_trials = []
    
    for i in range(0, len(eeg_data) - window_size + 1, stride):
        trials.append(eeg_data[i:i+window_size])
        joint_trials.append(joints_data[i:i+window_size])
    
    for i, (data, label) in enumerate(zip(trials, joint_trials)):
        dump_path = os.path.join(dump_folder, split, f"{split}_{subject_id}_{trial_id}_{i}.pkl")
        with open(dump_path, 'wb') as f:
            # 保存最后一个时间点的标签
            pickle.dump({"X": data, "y": label.iloc[-1]}, f)

def split_and_dump(data_dir, dump_folder):
    folder_paths = glob(os.path.join(data_dir, 'SL*-T*'))  # 获取 MoBI 文件夹中所有的 SLxx-Tyy 子文件夹

    for folder in folder_paths:
        eeg_file = os.path.join(folder, 'eeg.txt')  # EEG 数据文件路径
        joints_file = os.path.join(folder, 'joints.txt')  # 关节角度数据文件路径
        
        # 提取受试者序号和试验序号
        folder_name = os.path.basename(folder)
        subject_id = folder_name.split('-')[0]
        trial_id = folder_name.split('-')[1]

        (train_eeg_data, train_joints_data), (val_eeg_data, val_joints_data), (test_eeg_data, test_joints_data) = load_data_from_files(eeg_file, joints_file)
        
        # 保存数据
        save_data(train_eeg_data, train_joints_data, "train", dump_folder, subject_id, trial_id)
        save_data(val_eeg_data, val_joints_data, "val", dump_folder, subject_id, trial_id)
        save_data(test_eeg_data, test_joints_data, "test", dump_folder, subject_id, trial_id)


def main(data_dir, dump_folder):
    if not os.path.exists(dump_folder):
        os.makedirs(dump_folder)
        os.makedirs(os.path.join(dump_folder, "train"))
        os.makedirs(os.path.join(dump_folder, "val"))
        os.makedirs(os.path.join(dump_folder, "test"))

    split_and_dump(data_dir, dump_folder)

if __name__ == "__main__":
    data_dir = "/mnt/ssd/lingyus/MoBI"
    dump_folder = "/mnt/ssd/lingyus/MoBI/processed"
    main(data_dir, dump_folder)