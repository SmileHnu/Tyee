import os
import numpy as np
import mne
import pickle
import warnings
from multiprocessing import Pool
from sklearn.model_selection import train_test_split

channels = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 
            'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 
            'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 
            'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 
            'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 
            'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 
            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 
            'OZ', 'O2', 'CB2']

# 受试者和 session 的时间戳
session_timestamps = {
    1: {
        "start": [30, 132, 287, 555, 773, 982, 1271, 1628, 1730, 2025, 2227, 2435, 2667, 2932, 3204],
        "end": [102, 228, 524, 742, 920, 1240, 1568, 1697, 1994, 2166, 2401, 2607, 2901, 3172, 3359]
    },
    2: {
        "start": [30, 299, 548, 646, 836, 1000, 1091, 1392, 1657, 1809, 1966, 2186, 2333, 2490, 2741],
        "end": [267, 488, 614, 773, 967, 1059, 1331, 1622, 1777, 1908, 2153, 2302, 2428, 2709, 2817]
    },
    3: {
        "start": [30, 353, 478, 674, 825, 908, 1200, 1346, 1451, 1711, 2055, 2307, 2457, 2726, 2888],
        "end": [321, 418, 643, 764, 877, 1147, 1284, 1418, 1679, 1996, 2275, 2425, 2664, 2857, 3066]
    }
}

# 每个 session 对应的情绪标签
session_labels = {
    1: [4, 1, 3, 2, 0, 4, 1, 3, 2, 0, 4, 1, 3, 2, 0],
    2: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0],
    3: [2, 1, 3, 0, 4, 4, 0, 3, 2, 1, 3, 4, 1, 2, 0]
}

def load_data(file_path, session_id):
    try:
        eeg_raw = mne.io.read_raw_cnt(file_path, preload=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None
    
    # 删除不需要的通道
    useless_ch = ['M1', 'M2', 'VEO', 'HEO']
    eeg_raw.drop_channels(useless_ch)
    
    # 获取数据矩阵
    data_matrix = eeg_raw.get_data()
    
    # 获取时间戳
    start_seconds = session_timestamps[session_id]["start"]
    end_seconds = session_timestamps[session_id]["end"]
    
    # 分割数据
    trials = [data_matrix[:, start*1000:end*1000] for start, end in zip(start_seconds, end_seconds)]
    return trials

def preprocess_data(data):
    # 创建一个临时的 RawArray 对象
    info = mne.create_info(ch_names=channels, sfreq=1000, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # 带通滤波
    raw.filter(l_freq=0.1, h_freq=75.0)
    
    # 陷波滤波
    raw.notch_filter(50.0)
    
    # 重采样到200 Hz
    raw.resample(200, n_jobs=5)

    raw_data = raw.get_data(units='uV')
    return raw_data

def split_and_dump(params):
    data_dir, file_name, dump_folder = params
    parts = file_name.split('_')
    subject_id = int(parts[0])
    session_id = int(parts[1])
    file_path = os.path.join(data_dir, file_name)
    trials = load_data(file_path, session_id)
    
    if trials is None:
        return
    
    # 预处理数据
    trials = [preprocess_data(trial) for trial in trials]
    
    # 分割数据
    train_data, val_data, test_data, train_indices, val_indices, test_indices = split_data(trials, random_state=subject_id)
    
    # 保存数据
    for i, (data, idx) in enumerate(zip(train_data, train_indices)):
        label = session_labels[session_id][idx]
        for j in range(data.shape[1] // 2000):
            dump_path = os.path.join(dump_folder, "train", f"train_{subject_id}_{session_id}_{idx}_{j}.pkl")
            with open(dump_path, 'wb') as f:
                pickle.dump({"X": data[:, j * 2000 : (j + 1) * 2000], "y": label}, f)
    
    for i, (data, idx) in enumerate(zip(val_data, val_indices)):
        label = session_labels[session_id][idx]
        for j in range(data.shape[1] // 2000):
            dump_path = os.path.join(dump_folder, "val", f"val_{subject_id}_{session_id}_{idx}_{j}.pkl")
            with open(dump_path, 'wb') as f:
                pickle.dump({"X": data[:, j * 2000 : (j + 1) * 2000], "y": label}, f)
    
    for i, (data, idx) in enumerate(zip(test_data, test_indices)):
        label = session_labels[session_id][idx]
        for j in range(data.shape[1] // 2000):
            dump_path = os.path.join(dump_folder, "test", f"test_{subject_id}_{session_id}_{idx}_{j}.pkl")
            with open(dump_path, 'wb') as f:
                pickle.dump({"X": data[:, j * 2000 : (j + 1) * 2000], "y": label}, f)


def split_data(trials, train_size=5, val_size=5, test_size=5, random_state=None):
    # 确保试验数量为 15
    assert len(trials) == 15, "The number of trials must be 15."
    
    # 设置随机种子
    np.random.seed(random_state)
    
    # 创建试验索引
    trial_indices = list(range(len(trials)))
    
    # 按原顺序划分成三部分
    part1_indices = trial_indices[:train_size]
    part2_indices = trial_indices[train_size:train_size + val_size]
    part3_indices = trial_indices[train_size + val_size:]
    
    # 将三部分放入列表
    parts = [(part1_indices, trials[:train_size]), 
             (part2_indices, trials[train_size:train_size + val_size]), 
             (part3_indices, trials[train_size + val_size:])]
    
    # 打乱 parts 列表
    np.random.shuffle(parts)
    
    # 分别作为训练集、验证集和测试集
    (train_indices, train_data), (val_indices, val_data), (test_indices, test_data) = parts
    
    return train_data, val_data, test_data, train_indices, val_indices, test_indices

def main(data_dir):
    # 创建保存目录
    processed_dir = os.path.join(data_dir, "processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    train_dump_folder = os.path.join(processed_dir, "train")
    val_dump_folder = os.path.join(processed_dir, "val")
    test_dump_folder = os.path.join(processed_dir, "test")
    
    for folder in [train_dump_folder, val_dump_folder, test_dump_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    parameters = []
    for file_name in os.listdir(data_dir):
        if file_name.endswith('.cnt'):
            parameters.append([data_dir, file_name, processed_dir])
    
    # 使用多进程并行处理
    with Pool(processes=24) as pool:
        pool.map(split_and_dump, parameters)

if __name__ == "__main__":
    data_dir = '/home/lingyus/data/SEED-V/EEG_raw'
    main(data_dir)