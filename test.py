# from signal import PhysioSignalIO
import mne
import torch
import os
import numpy as np
from dataset.tuev_dataset import TUEVDataset
from dataset.tuab_dataset import TUABDataset
from dataset.seedv_dataset import SEEDVDataset
from dataset.bciciv2a_dataset import BCICIV2ADataset
from dataset.kaggleern_dataset import KaggleERNDataset
from dataset.sleepedfx_dataset import SleepEDFxDataset
from neurokit2.signal import signal_filter

# root_path = '/home/lingyus/data/TUEV/edf/train/aaaaaaar/aaaaaaar_00000001.edf'
# raw_data = mne.io.read_raw_edf(root_path)
# print(raw_data.info['ch_names'])

from dataset.transform import Compose, Filter, NotchFilter, PickChannels, OrderChannels, Resample, EA, UniToBiTransform
from dataset.transform import Scale, Cheby2Filter, Normalize
from dataset.constants import BIOT_CHANNELS
from dataset.split import DatasetSplitter
from torch.utils.data import DataLoader
import pandas as pd
import re
chOrder_standard = ['EEG FP1-REF', 'EEG FP2-REF', 'EEG F3-REF', 'EEG F4-REF', 'EEG C3-REF', 'EEG C4-REF', 'EEG P3-REF', 'EEG P4-REF', 'EEG O1-REF', 'EEG O2-REF', 'EEG F7-REF', \
                    'EEG F8-REF', 'EEG T3-REF', 'EEG T4-REF', 'EEG T5-REF', 'EEG T6-REF', 'EEG A1-REF', 'EEG A2-REF', 'EEG FZ-REF', 'EEG CZ-REF', 'EEG PZ-REF', 'EEG T1-REF', 'EEG T2-REF']
chOrder_standard = [name.split(' ')[-1].split('-')[0] for name in chOrder_standard]

seed_chan = ['FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 
            'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 
            'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 
            'FC2', 'FC4', 'FC6', 'FT8', 'T7', 'C5', 
            'C3', 'C1', 'CZ', 'C2', 'C4', 'C6', 'T8', 
            'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 
            'CP4', 'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 
            'PZ', 'P2', 'P4', 'P6', 'P8', 'PO7', 'PO5', 
            'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 
            'OZ', 'O2', 'CB2']
# # print(chOrder_standard)

# import pickle
# # # 指定文件路径
# file_path = '/home/lingyus/data/0_aaaaaaar_00000001'
# # file_path = '/home/lingyus/data/tuh_eeg_events/v2.0.1/edf/test/aaaaaaar_00000001-0.pkl'
# # file_path = '/home/lingyus/data/tuh_eeg_events/v2.0.1/edf/processed_labram_train/aaaaaaci_00000001-0.pkl'

# # 读取 .pkl 文件
# with open(file_path, 'rb') as file:
#     data = pickle.load(file)
# # 打印数据内容
# print(data['signals'].shape)

# raw = mne.io.read_raw_edf('/home/lingyus/data/test/sleep-cassette/SC4001E0-PSG.edf', preload=True)
# montage = mne.channels.make_standard_montage('standard_1020')
# raw.set_montage(montage, on_missing='ignore')

# annotation = mne.read_annotations('/home/lingyus/data/test/sleep-cassette/SC4001EC-Hypnogram.edf')
# raw.set_annotations(annotation, emit_warning=False)
# events, event_id = mne.events_from_annotations(
#     raw, chunk_duration=30.)

# if 'Sleep stage ?' in event_id.keys():
#     event_id.pop('Sleep stage ?')
# if 'Movement time' in event_id.keys():
#     event_id.pop('Movement time')

# tmax = 30. - 1. / raw.info['sfreq']
# epochs = mne.Epochs(raw=raw, events=events,
#                     event_id=event_id, tmin=0., tmax=tmax, baseline=None)

# epochs_data = epochs.get_data()
# chan = epochs.ch_names
# print(chan)
# epochs_label = []
# for epoch_annotation in epochs.get_annotations_per_epoch():
#     epochs_label.append(epoch_annotation[0][2])

# print(epochs_data.shape)


# import scipy.io
# # 读取 MATLAB 文件
# mat_file_path = '/home/lingyus/data/Data.mat'
# mat_data = scipy.io.loadmat(mat_file_path)
# # 打印文件中的所有变量
# print(mat_data.keys())
# # 假设你需要读取变量 'data'
# x_data = mat_data['x_data']
# y_data = mat_data['y_data']
# print(x_data[0])
# print(y_data.shape)

# train_files = ['aaaaadik', 'aaaaaeer', 'aaaaacpd', 'aaaaacxk', 'aaaaabvg', 'aaaaadpg', 
#                'aaaaadxh', 'aaaaafaj', 'aaaaacqc', 'aaaaaeqc', 'aaaaaaoy', 'aaaaadqq', 
#                'aaaaactm', 'aaaaadrd', 'aaaaabun', 'aaaaadvu', 'aaaaaafi', 'aaaaabth', 
#                'aaaaacko', 'aaaaadqg', 'aaaaaawm', 'aaaaabna', 'aaaaaavs', 'aaaaaayn', 
#                'aaaaafmb', 'aaaaacjh', 'aaaaadru', 'aaaaaahr', 'aaaaabgq', 'aaaaabgz', 
#                'aaaaacqe', 'aaaaaadx', 'aaaaaesm', 'aaaaadnh', 'aaaaabfu', 'aaaaaejk', 
#                'aaaaafjm', 'aaaaabea', 'aaaaaeio', 'aaaaadjn', 'aaaaabbu', 'aaaaabji', 
#                'aaaaacvq', 'aaaaadxl', 'aaaaaagq', 'aaaaadni', 'aaaaabxb', 'aaaaadvi', 
#                'aaaaabuu', 'aaaaadoa', 'aaaaadea', 'aaaaacxl', 'aaaaacer', 'aaaaaagx', 
#                'aaaaabrj', 'aaaaaehl', 'aaaaafgz', 'aaaaacon', 'aaaaaddp', 'aaaaaeoa', 
#                'aaaaaecu', 'aaaaaety', 'aaaaafgy', 'aaaaaaqg', 'aaaaaeqd', 'aaaaabix', 
#                'aaaaacxx', 'aaaaacjp', 'aaaaaajg', 'aaaaabxd', 'aaaaactf', 'aaaaabyj', 
#                'aaaaafai', 'aaaaadyv', 'aaaaafcu', 'aaaaabbw', 'aaaaabju', 'aaaaadxw', 
#                'aaaaaefs', 'aaaaafka', 'aaaaacid', 'aaaaabfp', 'aaaaacnc', 'aaaaabyi', 
#                'aaaaabos', 'aaaaadgr', 'aaaaaevv', 'aaaaabjc', 'aaaaaeyr', 'aaaaaaqz', 
#                'aaaaabqo', 'aaaaaffr', 'aaaaabhu', 'aaaaaecc', 'aaaaaadd', 'aaaaaatx', 
#                'aaaaacuq', 'aaaaadan', 'aaaaabgg', 'aaaaafmv', 'aaaaacof', 'aaaaabrr', 
#                'aaaaablf', 'aaaaadvc', 'aaaaaawx', 'aaaaacgq', 'aaaaadrh', 'aaaaaatm', 
#                'aaaaafku', 'aaaaacmk', 'aaaaaert', 'aaaaacnb', 'aaaaafeu', 'aaaaaasu', 
#                'aaaaaceh', 'aaaaaeaz', 'aaaaaavw', 'aaaaabop', 'aaaaaaug', 'aaaaadlc', 
#                'aaaaaezj', 'aaaaaboo', 'aaaaaaxh', 'aaaaacwx', 'aaaaaaie', 'aaaaadww', 
#                'aaaaaehv', 'aaaaabyt', 'aaaaaefj', 'aaaaaaet', 'aaaaabsy', 'aaaaadsh', 
#                'aaaaacwg', 'aaaaacse', 'aaaaaegi', 'aaaaaajc', 'aaaaaazl', 'aaaaabtr', 
#                'aaaaaewe', 'aaaaadsv', 'aaaaaeuh', 'aaaaacbl', 'aaaaadss', 'aaaaabjd', 
#                'aaaaadyg', 'aaaaabqq', 'aaaaacbc', 'aaaaabbn', 'aaaaaeqr', 'aaaaaexn', 
#                'aaaaacib', 'aaaaacwq', 'aaaaadbw', 'aaaaaeec', 'aaaaadat', 'aaaaacyw', 
#                'aaaaaeux', 'aaaaadey', 'aaaaaaar', 'aaaaabfa', 'aaaaacis', 'aaaaabym', 
#                'aaaaadxv', 'aaaaaegp', 'aaaaaewj', 'aaaaafdw', 'aaaaaadm', 'aaaaacqd', 
#                'aaaaafiv', 'aaaaadqt', 'aaaaadsd', 'aaaaaelw', 'aaaaacjf', 'aaaaaewi', 
#                'aaaaafff', 'aaaaaauq', 'aaaaaawk', 'aaaaaegl', 'aaaaadlg', 'aaaaadov', 
#                'aaaaaeyu', 'aaaaaeys', 'aaaaaetx', 'aaaaacrj', 'aaaaaaer', 'aaaaaems', 
#                'aaaaafop', 'aaaaafcs', 'aaaaadoo', 'aaaaaeie', 'aaaaachi', 'aaaaacww', 
#                'aaaaafef', 'aaaaabqd', 'aaaaabgo', 'aaaaacsw', 'aaaaabor', 'aaaaadul', 
#                'aaaaaabs', 'aaaaaadt', 'aaaaaamm', 'aaaaadsg', 'aaaaaeel', 'aaaaaffz', 
#                'aaaaaafx', 'aaaaabba', 'aaaaaawq', 'aaaaafoe', 'aaaaaaci', 'aaaaacvy', 
#                'aaaaadrg', 'aaaaacic', 'aaaaabdo', 'aaaaaaur', 'aaaaaetz', 'aaaaabel', 
#                'aaaaaads', 'aaaaaelg', 'aaaaafkp', 'aaaaaadk', 'aaaaaczk', 'aaaaaaff', 
#                'aaaaaboq', 'aaaaaeyz', 'aaaaaedd', 'aaaaaahp', 'aaaaacyp', 'aaaaaaxp', 
#                'aaaaaadg', 'aaaaaeic', 'aaaaaffh', 'aaaaabqs']

# val_files = ['aaaaacjz', 'aaaaadsr', 'aaaaabky', 'aaaaacur', 'aaaaaeex', 'aaaaadjd',
#             'aaaaaamc', 'aaaaaduk', 'aaaaacgy', 'aaaaaffm', 'aaaaaeyk', 'aaaaaexb',
#             'aaaaaadz', 'aaaaaaws', 'aaaaacpc', 'aaaaaevo', 'aaaaafmi', 'aaaaadio',
#             'aaaaaezm', 'aaaaafnb', 'aaaaaaof', 'aaaaafgm', 'aaaaadqz', 'aaaaaeri',
#             'aaaaaavn', 'aaaaaadn', 'aaaaaclp', 'aaaaafit', 'aaaaaahm', 'aaaaafmn',
#             'aaaaadty', 'aaaaaalp', 'aaaaaeur', 'aaaaablw', 'aaaaabsw', 'aaaaacks',
#             'aaaaadud', 'aaaaaesj', 'aaaaaduy', 'aaaaaako', 'aaaaabfy', 'aaaaabtw',
#             'aaaaadrv','aaaaafjb', 'aaaaabom', 'aaaaabgx', 'aaaaabtg', 'aaaaabxc',
#             'aaaaaeee', 'aaaaaeni', 'aaaaadst', 'aaaaachx', 'aaaaacml', 'aaaaaaxq',
#             'aaaaaccw', 'aaaaaamd', 'aaaaabcv', 'aaaaacqy']

# offline_transform ={
#     'eeg': Compose([
#             # PickChannels(channels=seed_chan),
#             # OrderChannels(order=seed_chan),
#             # Filter(l_freq=0.1, h_freq=75.0),
#             # Cheby2Filter(lowcut=4, highcut=40),
#             # NotchFilter(freqs=[50.0]),
#             # Filter(method='powerline', powerline=50),
#             # Scale(scale_factor=1e-6),
#             Normalize(method='min_max'),
#             # UniToBiTransform(target_channels=BIOT_CHANNELS),
#             # Resample(desired_sampling_rate=256.0, n_jobs=5)
#             ])
# } 

# after_trial = {
#     'eeg': Compose([
#             EA()
#         ])
# }

# id = 9
# id mean                   std
# 1 -0.004217920410355097   5.3160046572911135  
# 2 0.0009414219788173737   5.431030549625196 
# 3 -0.0025193005808739812  6.731697205541318   
# 4 -0.001424700059399228   4.943432844731517
# 5 -0.0014629550481563512  4.159579620276938 
# 6 -0.0054810448640211906  7.59114601601018 
# 7 0.00043640378624108816  4.757630552023434  
# 8 0.004634661521110368    9.090983761282917  
# 9 0.0005240510937519272   9.915030969774989
# root_path = '/home/lingyus/data/TUEV/edf/eval/'
# io_path = "/home/lingyus/data/TUEV/edf/processed_eval"
# train_dataset = KaggleERNDataset(root_path=f'/home/lingyus/data/KaggleERN/train', 
#                       label_path='/home/lingyus/data/KaggleERN/TrainLabels.csv',
#                       io_path=f"/home/lingyus/data/KaggleERN/train/processed", 
#                       io_mode='pickle', 
#                       num_worker=8, 
#                       # chunk_size=1000,
#                       # offset=500,
#                       offline_transform=offline_transform)

# test_dataset = KaggleERNDataset(root_path='/home/lingyus/data/SEED-V/EEG_raw', 
#                                 label_path='/home/lingyus/data/KaggleERN/true_labels.csv',
#                       io_path="/home/lingyus/data/KaggleERN/test/processed", 
#                       studies= ['cassette',],
#                       io_mode='pickle', 
#                       num_worker=8, 
#                       # chunk_size=1000,
#                       # offset=500,
#                       offline_transform=offline_transform,
#                     # after_trial=after_trial
#                     )


# import wfdb
# from dataset.split import DatasetSplitter
from dataset.transform import WaveletTransform
from dataset.mit_bih_dataset import MITBIHDataset
# dataset = MITBIHDataset(root_path='/home/lingyus/data/mit-bih-arrhythmia-database-1.0.0',
#                         io_path='/home/lingyus/data/mit-bih-arrhythmia-database-1.0.0/processed_wavelet',
#                         io_mode='pickle',
#                         pre_offset=100,
#                         post_offset=140,
#                         before_trial={'ecg':Compose([
#                             WaveletTransform(wavelet='db5', level=9),
#                         ])},
#                         num_worker=8,)
# file_name =['100', '101', '102', '103', '104', '105', '106', '107',
#             '108', '109', '111', '112', '113', '114', '115', '116',
#             '117', '118', '119', '121', '122', '123', '124', '200',
#             '201', '202', '203', '205', '207', '208', '209', '210',
#             '212', '213', '214', '215', '217', '219', '220', '221',
#             '222', '223', '228', '230', '231', '232', '233', '234']

# print(dataset[0]['ecg']['signals'])
# harmonics = np.array([i * 50 for i in range(1, (1000 // 2) // 50)])
# print(harmonics)
# print(np.logspace(np.log10(40), np.log10(300), 40))
# for file in file_name:
#     labels = []
#     filtered_indices = dataset.info[dataset.info['subject_id'] == int(file)].index
#     for i in filtered_indices:
#         labels.append(dataset[i]['label'])
        
#     label_count = pd.Series(labels).value_counts()
#     print(f'File {file} label count:')
#     print(label_count)
# labels = [dataset[i]['label'] for i in range(len(dataset))]
# label_count = pd.Series(labels).value_counts()
# print(label_count)
# 0    90631
# 2     7236
# 1     2781
# 3      803
# splitter = DatasetSplitter(train_dataset=dataset)
# split = splitter.split(method='kfold', split_by='clip_id', stratified=False, n_splits=5, shuffle=True)
# for train_dataset, val_dataset, test_dataset in split:
#     # 统计训练集和验证集的标签分布
#     # dataset没有get_label()方法，所以需要自己从dataset中提取标签
#     train_labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
#     val_labels = [val_dataset[i]['label'] for i in range(len(val_dataset))]
#     train_label_count = pd.Series(train_labels).value_counts()
#     val_label_count = pd.Series(val_labels).value_counts()
#     print(train_label_count)
#     print(val_label_count)

# print(dataset[0])
# print(len(dataset))
# print(dataset[12]['ecg']['signals'].shape)
# print(dataset[12]['ecg']['channels'])

# root_path = '/home/lingyus/data/mit-bih-atrial-fibrillation-database-1.0.0'
# data_files = os.listdir(root_path)
# data_files = np.array(data_files)
# integers = [int(re.search(r'\d+', item).group()) for item in data_files if re.search(r'\d+', item)]
# file_int = set(integers)
# file_int = list(file_int)
# file_int.remove(256)
# file_int.remove(5121)
# file_list = [os.path.join(root_path, str(item)) for item in file_int]
# # print(file_list)
# aami_mapping = {
#     'N': 'N', 'L': 'N', 'R': 'N', 'e': 'N', 'j': 'N',  # N 类（正常）
#     'A': 'S', 'a': 'S', 'J': 'S', 'S': 'S',  # S 类（房性心律失常）
#     'V': 'V', 'E': 'V',  # V 类（室性心律失常）
#     'F': 'F'  # F 类（融合搏动）
# }
# symbols = np.array(['N','L','R','A','a','J','S','V','F','e','j','E'])
# for file_name in file_list:
#     record = wfdb.rdsamp(file_name)
#     # data = record[0].transpose()
#     print(record)
#     annotation = wfdb.rdann(file_name,'atr')
#     print(annotation.symbol)
#     print(annotation.sample)
    # Index = np.isin(annotation.symbol, symbols)
    # labels = np.array(annotation.symbol)[Index]
    # R_location = np.array(annotation.sample)[Index]
    # print(len(labels))
    # print(len(R_location))
    # print(labels)
    # print(R_location)
    # gt = np.zeros(len(labels), dtype='float')
    # mapped_labels = [(aami_mapping.get(label, 'UNKNOWN')) for label in labels]
    # # print(mapped_labels)
    # N=0
    # S=0
    # V=0
    # F=0
    # ecg = data[:,R_location[1]-100:R_location[1]+200]
    # print(ecg)
    # print(ecg.shape)
    # print(mapped_labels[1])
    # for index, item in enumerate(mapped_labels):
    #     if index<5:
    #         continue
    #     ecg = data[:,R_location[index]-100:R_location[index]+200]
    #     label = item
    #     print(ecg)
    #     print(label)
    #     print(ecg.shape)
    #     if item == 'UNKNOWN':
    #         print(f'Unknown label: {labels[index]}')
    #     if item == 'N':
    #         N += 1
    #     elif item == 'S':
    #         S += 1
    #     elif item == 'V':
    #         V += 1
    #     elif item == 'F':
    #         F += 1
    # print(f'N: {N}, S: {S}, V: {V}, F: {F}')
    # print(record)
    # print(labels)
    # print(gt)
    # print(R_location)



# test /home/lingyus/data/NinaproDB5/s1/S1_E1_A1.mat
from dataset.ninapro_db5_dataset import NinaproDB5Dataset

dataset = NinaproDB5Dataset(root_path='/home/lingyus/data/test',
                            io_path='/home/lingyus/data/test/processed',
                            # io_mode='pickle',
                            offline_transform=None,
                            num_worker=8)
print(dataset[0]['emg']['signals'].shape)

# from imblearn.over_sampling import SMOTE
# import numpy as np

# # 生成示例数据
# X = np.random.rand(20, 2, 4)  # 100 个样本，每个 10 维
# print(X.shape)
# y = np.concatenate([np.zeros(12), np.ones(8)])  # 类别 0: 80 个，类别 1: 20 个（不平衡）

# # 过采样
# smote = SMOTE(sampling_strategy='auto', random_state=42)
# X_resampled, y_resampled = smote.fit_resample(X, y)
# print("过采样后数据形状:", X_resampled)
# print("过采样后标签形状:", y_resampled)
# print("原始类别分布:", np.bincount(y.astype(int)))
# print("过采样后类别分布:", np.bincount(y_resampled.astype(int)))
# import scipy.io
# from scipy.ndimage import label
# # 读取 DB5 的 mat 文件
# data = scipy.io.loadmat("/home/lingyus/data/NinaproDB5/s1/S1_E2_A1.mat")  # 示例文件名

# # 提取不同的数据
# emg = data['emg']          # sEMG 信号 (样本数 × 12)
# glove = data['glove']      # 手指关节角度 (样本数 × 22)
# acc = data['acc']          # IMU 传感器数据 (样本数 × 6)
# stimulus = data['stimulus']  # 手势标签 (样本数 × 1)
# restimulus = data['restimulus']  # 重复手势标签 (样本数 × 1)
# frequency = data['frequency']  # 采样频率 (1 × 1)
# window_size = 40  # 200 ms 对应 40 个采样点 (200ms × 2000Hz)
# stride = 8  # 每次滑动 4 个采样点
# print(data.keys())
# print(emg.shape)
# print(glove.shape)
# print(restimulus.shape)
# emg_windows = np.array([emg[i:i+window_size] for i in range(0, len(emg)-window_size, stride)])
# 识别连续的手势段
# mask = restimulus > 0  # 过滤掉0，得到二值化 mask
# labeled_array, num_segments = label(mask)  # 识别连续的非0片段

# all_windows = []
# all_labels = []

# for seg_id in range(1, num_segments + 1):  # 遍历每个手势段
#     segment_indices = np.where(labeled_array == seg_id)[0]  # 该段的索引
#     segment_emg = emg[segment_indices]  # 取出对应的 sEMG 片段
#     segment_label = restimulus[segment_indices]  # 取出对应的标签

#     unique_labels = np.unique(segment_label)
#     # print(unique_labels)
#     if len(unique_labels) > 1:
#         raise ValueError(f"Segment {seg_id} 存在多个不同标签: {unique_labels}")

#     hand_gesture = unique_labels[0]  # 该段唯一的手势标签

#     # 滑动窗口提取样本
#     for start in range(0, len(segment_indices) - window_size + 1, stride):
#         window = segment_emg[start:start + window_size]  # 取窗口内的信号
#         all_windows.append(window)
#         all_labels.append(hand_gesture)  # 分配该段手势标签


# test /home/lingyus/data/BCICIV4/sub1_comp.mat
# from dataset.bciciv4_dataset import BCICIV4Dataset
# from dataset.transform import CAR,TFR
# offline_transform ={
#     'ecog': Compose([
#         Normalize(method='zscore_per_channel',epsilon=0),
#         CAR(),
#         Filter(l_freq=40, h_freq=300),
#         NotchFilter(freqs=[50, 100, 150, 200, 250, 300, 350, 400, 450]),
#         TFR(freqs = np.logspace(np.log10(30), np.log10(400), 40),output='power',n_jobs=6),
#     ])}
            
# dataset = BCICIV4Dataset(root_path='/home/lingyus/data/BCICIV4',
#                         io_path='/mnt/ssd/lingyus/BCICIV4_processed',
#                         io_mode='pickle',
#                         chunk_size= 256,
#                         overlap=40, 
#                         offline_transform=offline_transform,
#                         num_worker=8)

# splitter = DatasetSplitter(train_dataset=dataset)
# split = splitter.split(method='loso', split_by='subject_id', stratified=False, n_splits=5, shuffle=True)
# train_dataset, val_dataset, test_dataset = split[0]
# print(train_dataset[0]['label'])
# for train_dataset, val_dataset, test_dataset in split:
#     # 统计训练集和验证集的标签分布
#     # dataset没有get_label()方法，所以需要自己从dataset中提取标签
#     train_labels = [train_dataset[i]['label'] for i in range(len(train_dataset))]
#     val_labels = [val_dataset[i]['label'] for i in range(len(val_dataset))]
#     train_label_count = pd.Series(train_labels).value_counts()
#     val_label_count = pd.Series(val_labels).value_counts()
#     print(train_label_count)
#     print(val_label_count)
# print(dataset[0]['ecog']['signals'].shape)
# print(dataset[0]['ecog']['signals'])
# print(dataset[0]['label'])
# import scipy.io

# file_path = '/home/lingyus/data/BCICIV4/sub1_comp.mat'
# data = scipy.io.loadmat(file_path)
# test_label = scipy.io.loadmat(file_path.replace('comp.mat','testlabels.mat'))

# train_data = data['train_data'].T
# train_dg = data['train_dg'].T
# test_data = data['test_data'].T
# test_dg = test_label['test_dg'].T
# print(train_data)
# print(test_data)
# print(train_data[:,0:40])
# print(test_data[:,0:40])
# print(test_data[:,0:40].shape)
# print(train_dg[:,39:40])
# print(test_dg[:,39:40])
# print(train_data.shape)
# print(train_dg.shape)
# print(test_data.shape)
# print(test_dg.shape)

# 测试指标

# from utils import MetricEvaluator
# # 构造测试数据 (batch, 输出层数量, 输出结果)
# true_values = torch.tensor([
#     [[1.0, 2.0], [3.0, 4.0]],
#     [[5.0, 6.0], [7.0, 8.0]]
# ])  # shape: (2, 2, 2)

# predicted_values = torch.tensor([
#     [[1.1, 1.9], [3.2, 3.8]],
#     [[4.9, 6.1], [7.1, 7.9]]
# ])  # shape: (2, 2, 2)

# # 将测试数据包装成结果字典
# result = {"label": true_values, "output": predicted_values}
# # print(true_values.type)

# # 初始化 MetricEvaluator，传入需要测试的指标名称
# evaluator = MetricEvaluator(metric_list=["mse", "mae"])

# # 更新指标，模拟多次 batch 的结果
# for _ in range(3):  # 假设有 3 个 batch
#     evaluator.update_metrics(result)

# # 计算所有指标
# metrics_result = evaluator.calculate_metrics()

# # 打印结果
# print("Metrics Result:", metrics_result)