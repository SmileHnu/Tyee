# tyee.dataset.transform

`tyee.dataset.transform`模块提供丰富的信号预处理方法可帮助用户提取特征、构建信号表示。

| 变换方法名称                               | 功能作用                                                     |
| ------------------------------------------ | ------------------------------------------------------------ |
| [`BaseTransform`](#basetransform)         | 所有变换类的基类，定义了变换操作的基本接口和使用方式。       |
| [`CommonAverageRef`](#commonaverageref)   | 对信号的每个采样点进行共同平均参考（CAR）或共同中位数参考（CMR）基线校正。 |
| [`PickChannels`](#pickchannels)           | 从信号中选择指定的通道。                                     |
| [`OrderChannels`](#orderchannels)         | 按照指定的顺序重排信号通道，对于缺失的通道可以进行填充。     |
| [`ToIndexChannels`](#toindexchannels)     | 将信号数据中的通道名称列表根据给定的主列表转换为对应的索引。 |
| [`UniToBiTransform`](#unitobitransform)   | 将单极信号通过计算指定通道对之间的差值转换为双极信号。       |
| [`Crop`](#crop)                           | 通用裁剪变换，可沿指定轴从信号的左侧（起始）和/或右侧（末尾）裁剪指定数量的点。 |
| [`Filter`](#filter)                       | 应用滤波器（如带通、低通、高通）到信号数据，基于 `mne.filter.filter_data`。 |
| [`NotchFilter`](#notchfilter)             | 应用陷波滤波器以移除信号数据中的特定频率成分（如工频噪声），基于 `mne.filter.notch_filter`。 |
| [`Cheby2Filter`](#cheby2filter)           | 应用切比雪夫II型数字或模拟滤波器到信号数据，基于 `scipy.signal.cheby2`。 |
| [`Detrend`](#detrend)                     | 移除信号数据中沿指定轴的线性趋势，基于 `scipy.signal.detrend`。 |
| [`Compose`](#compose)                     | 将一系列变换组合起来，按顺序应用于输入数据。                 |
| [`ForEach`](#foreach)                     | 沿数据的第一个维度（如epochs或trials），对每个片段分别应用一系列指定的子变换。 |
| [`Lambda`](#lambda)                       | 应用用户定义的lambda函数或任何可调用对象到信号字典中的'data'字段，以执行自定义操作。 |
| [`Mapping`](#mapping)                     | 对信号字典中 'data' 字段的值应用字典映射。如果 'data' 是标量则直接映射，如果是数组或可迭代对象则逐个元素映射。 |
| [`Concat`](#concat)                       | 将来自多个输入信号字典的 'data' 数组沿指定轴连接起来，并尝试合并通道名和保留频率信息。 |
| [`Stack`](#stack)                         | 将来自多个输入信号字典的 'data' 数组沿新轴堆叠起来。         |
| [`ZScoreNormalize`](#zscorenormalize)     | 对信号数据执行Z-score标准化（(x - mean) / std）。            |
| [`MinMaxNormalize`](#minmaxnormalize)     | 对信号数据执行最小-最大标准化，将数据缩放到一个指定范围（通常是0到1）。 |
| [`QuantileNormalize`](#quantilenormalize) | 对信号数据执行分位数标准化，通过将数据除以其绝对值的指定分位数进行归一化。 |
| [`RobustNormalize`](#robustnormalize)     | 执行稳健标准化 `(x - median) / IQR`，对异常值具有抵抗性。    |
| [`Baseline`](#baseline)                   | 对信号进行基线校正，减去指定区间的均值。                     |
| [`Mean`](#mean)                           | 计算信号字典中 'data' 字段沿指定轴的均值，并用计算得到的均值替换原始数据。 |
| [`OneHotEncode`](#onehotencode)           | 将输入数据（通常是分类标签）转换为独热编码格式。             |
| [`Pad`](#pad)                             | 对信号字典中的 'data' 数组沿指定轴进行填充。                 |
| [`Resample`](#resample)                   | 使用MNE库将信号数据重采样到指定的目标频率。                  |
| [`Downsample`](#downsample)               | 通过选取每N个样本的方式，将信号数据从当前频率下采样到指定的目标频率。 |
| [`Interpolate`](#interpolate)             | 使用插值方法（如线性、立方）将信号数据从当前频率上采样到指定的目标频率。 |
| [`Reshape`](#reshape)                     | 将信号字典中的 'data' 数组重塑为指定的目标形状。             |
| [`Transpose`](#transpose)                 | 转置信号字典中的 'data' 数组的轴。                           |
| [`Squeeze`](#squeeze)                     | 从信号字典中 'data' 数组的形状中移除单维度条目。             |
| [`ExpandDims`](#expanddims)               | 在信号字典中 'data' 数组的指定位置插入一个新的轴（维度）。   |
| [`Insert`](#insert)                       | 在信号字典 'data' 数组的指定轴的给定索引处插入指定值。       |
| [`ImageResize`](#imageresize)             | 将图像数据（通常在 'data' 字段）调整为指定的尺寸，使用 `torchvision.transforms.Resize`。 |
| [`Scale`](#scale)                         | 通过乘以一个指定的缩放因子来对信号数据进行数值缩放。         |
| [`Offset`](#offset)                       | 通过加上一个指定的偏移量来对信号数据进行数值偏移。           |
| [`Round`](#round)                         | 对信号数据中的数值进行四舍五入到最接近的整数。               |
| [`Log`](#log)                             | 对信号数据应用自然对数变换，通常在数据加上一个小的epsilon以避免log(0)。 |
| [`Select`](#select)                       | 根据指定的单个键或键列表，从输入字典中选择一部分条目。       |
| [`SlideWindow`](#slidewindow)             | 沿信号数据的指定轴计算滑动窗口的开始和结束索引。             |
| [`WindowExtract`](#windowextract)         | 根据滑动窗口信息从数据中提取窗口片段并堆叠。                 |
| [`CWTSpectrum`](#cwtspectrum)             | 使用Morlet小波通过连续小波变换（CWT）计算信号的时频表示。    |
| [`DWTSpectrum`](#dwtspectrum)             | 使用离散小波变换（DWT）计算信号的小波系数。                  |
| [`FFTSpectrum`](#fftspectrum)             | 使用快速傅里叶变换（FFT）计算信号的频域幅度谱。              |
| [`ToImage`](#toimage)                     | 将输入的信号数据（通常为2D）转换为图像表示，涉及归一化、颜色映射、调整大小等步骤。 |
| [`ToTensor`](#totensor)                   | 将NumPy数组格式的 'data' 转换为PyTorch张量。                 |
| [`ToTensorFloat32`](#totensorfloat32)     | 将 'data' 转换为PyTorch张量并确保其数据类型为float32。       |
| [`ToTensorFloat16`](#totensorfloat16)     | 将 'data' 转换为PyTorch张量并确保其数据类型为float16 (half)。 |
| [`ToTensorInt64`](#totensorint64)         | 将 'data' 转换为PyTorch张量并确保其数据类型为int64 (long)。  |
| [`ToNumpy`](#tonumpy)                     | 将PyTorch张量格式的 'data' 转换为NumPy数组。                 |
| [`ToNumpyFloat64`](#tonumpyfloat64)       | 将 'data' 转换为NumPy数组并确保其数据类型为float64。         |
| [`ToNumpyFloat32`](#tonumpyfloat32)       | 将 'data' 转换为NumPy数组并确保其数据类型为float32。         |
| [`ToNumpyFloat16`](#tonumpyfloat16)       | 将 'data' 转换为NumPy数组并确保其数据类型为float16。         |
| [`ToNumpyInt64`](#tonumpyint64)           | 将 'data' 转换为NumPy数组并确保其数据类型为int64。           |
| [`ToNumpyInt32`](#tonumpyint32)           | 将 'data' 转换为NumPy数组并确保其数据类型为int32。           |



## BaseTransform

我们提供`BaseTransform`基类，定义transform操作的接口和使用方式

~~~python
class BaseTransform(source:str| list[str] = None, target: str = None)
~~~

**参数**

- `source(str| list[str])`：指定该操作所针对的信号类型字段或字段列表
- `target(str)`: 指定处理后结果保存的字段

**方法**

~~~python
transform(result:dict)->dict
~~~

参数：

- `result(dict)`: 包含需要处理的信号字段的字典，每个字段可能包含以下信息：
  - `data`: 信号数据
  - `freq`：采样率
  - `channels`：通道列表
  - `info`：信号的其他描述信息，比如每个样本的起止索引列表等

我们基于`BaseTransform`实现了一些预处理操作，此外用户也可基于`BaseTransform`进行自定义新的预处理操作。

[`返回顶部`](#tyeedatasettransform)

## CommonAverageRef

~~~python
class CommonAverageRef(axis:int=0, mode:str = 'median', source:str| list[str] = None, target: str = None)
~~~

对信号的每个采样点做基线参考

**参数**

- `axis(int)`：决定对哪一维度做，通常是对通道维
- `mode(str)`：参考值选择，提供'median'(中位数)，'mean'(均值)。可扩展

**使用举例**

~~~python
results = {
    'eeg': {
        'data': np.random.randn(32,128)
        'freq': 200.0
        'channels': [i for i in range(0,32)]
    }
}
t= CommonAverageRef(axis=0, mode='median', source='eeg', target='eeg')
results = t(results)
~~~

[`返回顶部`](#tyeedatasettransform)

## PickChannels

~~~python
class PickChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], source: str = None, target: str = None):
~~~

从输入信号中选择指定的通道。如果在输入信号中找不到为拾取指定的通道，则会引发 `KeyError`。

**参数**

-   `channels (Union[str, List[str]])`: 要选择的通道名称列表。或者，这可以是一个字符串键，用于动态导入通道名称列表（例如，从 `dataset.constants`）。
-   `source (str, optional)`: 输入字典中包含待转换信号数据的键。默认为 `None`。
-   `target (str, optional)`: 输出字典中用于存储转换后信号数据的键。如果为 `None` 或与 `source` 相同，则原始信号数据将被覆盖。默认为 `None`。

**使用样例**

~~~python
# 假设 'results' 是一个字典，例如:
# results = {
#     'eeg': {
#         'data': np.random.randn(4, 128), # 示例: 4 个通道
#         'channels': ['Fz', 'Cz', 'Pz', 'Oz'],
#         'freq': 200.0
#     }
# }
# 定义要选择的通道列表 (确保这些通道存在于 results['eeg']['channels'] 中)
# 例如, 如果 results['eeg']['channels'] 是 ['Fz', 'Cz', 'Pz', 'Oz']
# 并且我们想选择 ['Fz', 'Pz']
pick_these_channels = ['Fz', 'Pz'] # 确保这些是输入中的有效通道
# 初始化 PickChannels，使其作用于 'eeg' 并存储在 'eeg_picked'
t = PickChannels(channels=pick_these_channels, source='eeg', target='eeg_picked')
# 应用变换
processed_results = t(results)
# processed_results['eeg_picked']['data'] 的形状将是 (2, 128)
# processed_results['eeg_picked']['channels'] 将是 ['Fz', 'Pz']
~~~

---

[`返回顶部`](#tyeedatasettransform)

## OrderChannels 

~~~python
class OrderChannels(BaseTransform):
    def __init__(self, order: Union[str, List[str]], padding_value: float = 0, source: str = None, target: str = None):
~~~

根据指定的 `order` 重排输入信号的通道顺序。如果 `order` 中指定的通道在输入信号中不存在，会抛出 `KeyError`。输入信号中存在但不在 `order` 列表中的通道将被丢弃。此变换主要用于统一不同数据集的通道顺序。

**参数**

-   `order (Union[str, List[str]])`: 指定期望通道名称顺序的列表。这也可以是用于动态导入通道顺序列表的字符串键。
-   `padding_value (float, optional)`: 保留参数（当前实现中未使用）。默认为 `0`。
-   `source (str, optional)`: 输入字典中待转换信号的键。默认为 `None`。
-   `target (str, optional)`: 输出字典中转换后信号的键。默认为 `None`。


**使用样例**

~~~python
# 假设 'results' 是一个字典，例如:
# results = {
#     'eeg': {
#         'data': np.random.randn(3, 100), # Cz, Fz, Pz 的数据
#         'channels': ['Cz', 'Fz', 'Pz'],   # 原始顺序
#         'freq': 100.0
#     }
# }
# 定义期望的通道顺序（重排序现有通道）
desired_order = ['Fz', 'Cz', 'Pz'] # 重排序: Fz和Cz交换位置，Pz保持在最后
# 初始化 OrderChannels
t = OrderChannels(order=desired_order, padding_value=np.nan, source='eeg', target='eeg_ordered')
# 应用变换
processed_results = t(results)
# processed_results['eeg_ordered']['data'] 的形状将是 (3, 100)
# processed_results['eeg_ordered']['channels'] 将是 ['Fz', 'Cz', 'Pz']
# 数据行将按照新的通道顺序重新排列：
# 第0行：原来Fz的数据（原来是第1行）
# 第1行：原来Cz的数据（原来是第0行）  
# 第2行：原来Pz的数据（保持第2行）
~~~

---

[`返回顶部`](#tyeedatasettransform)

## ToIndexChannels

~~~python
class ToIndexChannels(BaseTransform):
    def __init__(self, channels: Union[str, List[str]], strict_mode: bool = False, source: str = None, target: str = None):
~~~

根据提供的主 `channels` 列表，将输入信号 `channels` 列表中的通道名称转换为其对应的整数索引。根据提供的 Python 代码，此转换本身不修改 `data` 数组；仅更新 `channels` 字段。

**参数**

-   `channels (Union[str, List[str]])`: 定义从名称到索引（0基）映射的主通道名称列表。这也可以是用于动态导入此主列表的字符串键。
-   `strict_mode (bool, optional)`: 如果为 `True`，则输入信号 `channels` 列表中存在的所有通道名称*必须*存在于主 `channels` 列表中；否则将引发错误。如果为 `False`，则在主列表中未找到的输入通道名称将被静默忽略。默认为 `False`。
-   `source (str, optional)`: 输入字典中待转换信号的键。默认为 `None`。
-   `target (str, optional)`: 输出字典中转换后信号的键。默认为 `None`。

**使用样例**

~~~python
# 假设 'master_channel_list' 已定义, 例如:
# master_channel_list = ['Fp1', 'Fp2', 'Fz', 'Cz', 'Pz', 'Oz']
# 假设 'results' 是一个字典，例如:
# results = {
#     'eeg': {
#         'data': np.random.randn(3, 50),   # Cz, Fz, Oz 的数据
#         'channels': ['Cz', 'Fz', 'Oz'],   # 待转换的通道
#         'freq': 100.0
#     }
# }
# 初始化 ToIndexChannels
t = ToIndexChannels(channels=master_channel_list, strict_mode=False, source='eeg', target='eeg_indexed')
# 应用变换
processed_results = t(results)
# processed_results['eeg_indexed']['channels'] 将是 [3, 2, 5]
# (master_channel_list 中 Cz, Fz, Oz 的索引)
# 'data' 数组保持不变。
~~~

[`返回顶部`](#tyeedatasettransform)

## Compose

~~~python
class Compose(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], source: Optional[str] = None, target: Optional[str] = None):
~~~

一个组合其他变换列表的变换，它会按顺序应用这些变换。
`source` 和 `target` 参数用于 `Compose` 实例本身，定义了当它作用于一个结果字典时，从哪里读取数据以及向哪里写入数据。
在 `transforms` 列表中提供的子变换在初始化时**必须不设置 `source` 或 `target` 参数**，因为它们直接操作流经 `Compose` 管道的数据。尝试使用已设置 `source` 或 `target` 的子变换来初始化 `Compose` 将会引发 `ValueError`（根据您提供的实现）。

**参数**

-   `transforms (List[BaseTransform])`: 一个按顺序应用的变换实例列表。此列表中的每个变换在初始化时都不应设置 `source` 或 `target`。
-   `source (Optional[str])`: 输入字典中的键，`Compose` 变换从此键读取其初始数据。默认为 `None`。
-   `target (Optional[str])`: 输出字典中的键，`Compose` 变换在此键写入最终处理后的数据。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**内部 `transform` 逻辑说明:**
`Compose` 类的 `transform` 方法（根据您提供的代码）会遍历其子变换。对于每个子变换 `t`，它实际上会执行 `result = t(result)`（因为根据 `__init__` 中的检查，`t.source` 和 `t.target` 均为 `None`，这将导致 `current_data_item = t_sub({'data': current_data_item})['data'] if t_sub.source or t_sub.target else t_sub(current_data_item)` 条件语句的 `else` 分支被执行）。这依赖于子变换的 `__call__` 方法能够直接处理原始数据（在此上下文中，通常是信号字典）。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, PickChannels, CommonAverageRef, Compose) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import PickChannels, CommonAverageRef, Compose

# 1. 准备一个包含初始信号数据的示例 'results' 字典
results = {
    'raw_eeg': {
        'data': np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]]), # 4 通道, 3 时间点
        'channels': ['CH1', 'CH2', 'CH3', 'CH4'],
        'freq': 100.0
    }
}

# 2. 实例化子变换 (初始化时不带 source/target)
pick_specific_channels = PickChannels(channels=['CH1', 'CH3', 'CH4'])
apply_car_transform = CommonAverageRef(axis=0, mode='mean')

# 3. 实例化 Compose 变换
#    'source' 和 'target' 用于 Compose 实例本身。
pipeline = Compose(
    transforms=[
        pick_specific_channels,
        apply_car_transform
    ],
    source='raw_eeg',
    target='processed_eeg'
)

# 4. 应用组合变换
processed_results = pipeline(results)

# 5. 'processed_results' 现在将包含带有转换后数据的 'processed_eeg' 键。
# processed_results['processed_eeg'] 的示例内容:
# {
#   'data': np.array([[-5., -5., -5.], [ 1.,  1.,  1.], [ 4.,  4.,  4.]]),
#   'channels': ['CH1', 'CH3', 'CH4'],
#   'freq': 100.0
# }
# print(processed_results['processed_eeg'])
~~~

[`返回顶部`](#tyeedatasettransform)

## UniToBiTransform

~~~python
class UniToBiTransform(BaseTransform):
    def __init__(self, target_channels: Union[str, List[str]], source: str = None, target: str = None):
~~~

将单极信号转换为双极信号。这是通过计算 `target_channels` 列表中指定的相邻通道之间的差值来实现的。`target_channels` 中的每个字符串应定义一对单极通道，这两个通道将被转换为一个双极通道（例如，"Fp1-F7"）。如果在输入信号中找不到任何指定的单极通道，则会引发 `ValueError`。

**参数**

-   `target_channels (Union[str, List[str]])`: 一个字符串列表，其中每个字符串以 "通道A-通道B" 的格式指定一个双极通道对。或者，这可以是一个字符串键，用于动态导入此类通道对字符串的列表（例如，从 `dataset.constants`）。
-   `source (str, optional)`: 输入字典中的键，该键包含单极信号数据（表现为一个包含 'data'、'channels'以及可选的 'freq' 的信号字典）。默认为 `None`。
-   `target (str, optional)`: 输出字典中的键，转换后的双极信号数据（表现为一个新的信号字典）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, UniToBiTransform) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import UniToBiTransform

# 1. 准备一个包含初始单极信号数据的示例 'results' 字典
results = {
    'unipolar_eeg': {
        'data': np.array([[1.0, 1.5, 2.0], [0.5, 0.8, 1.2], [2.0, 2.2, 2.5], [1.0, 1.0, 1.0]]),
        'channels': ['Fp1', 'F7', 'Fp2', 'F8'], # 单极通道
        'freq': 250.0
    }
}

# 2. 定义目标双极通道对
bipolar_pairs = ["Fp1-F7", "Fp2-F8"]

# 3. 实例化 UniToBiTransform
#    'source' 和 'target' 用于 UniToBiTransform 实例本身。
transformer = UniToBiTransform(
    target_channels=bipolar_pairs,
    source='unipolar_eeg',
    target='bipolar_eeg'
)

# 4. 应用变换
processed_results = transformer(results)

# 5. 'processed_results' 现在将包含 'bipolar_eeg' 键。
#    processed_results['bipolar_eeg']['data'] 将包含差值:
#    Fp1-F7: [1.0-0.5, 1.5-0.8, 2.0-1.2] = [0.5, 0.7, 0.8]
#    Fp2-F8: [2.0-1.0, 2.2-1.0, 2.5-1.0] = [1.0, 1.2, 1.5]
#    processed_results['bipolar_eeg']['channels'] 将是 ["Fp1-F7", "Fp2-F8"]。
# processed_results['bipolar_eeg'] 的示例内容:
# {
#   'data': np.array([[0.5, 0.7, 0.8], [1.0, 1.2, 1.5]]),
#   'channels': ["Fp1-F7", "Fp2-F8"],
#   'freq': 250.0
# }
# print(processed_results['bipolar_eeg'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Crop

~~~python
class Crop(BaseTransform):
    def __init__(self, crop_left=0, crop_right=0, axis=-1, source=None, target=None):
~~~

一个通用的裁剪 transform，可指定裁剪起止位置或左右裁剪点数，适用于任意信号。它能沿给定轴（通常是时间轴）从信号的左侧（起始）和/或右侧（末尾）裁剪指定数量的点。

**参数**

-   `crop_left (int)`: 从信号左侧（起始位置）裁剪掉的点数。默认为 `0`。
-   `crop_right (int)`: 从信号右侧（末尾位置）裁剪掉的点数。如果为 `0` 或负数，则不从右侧进行裁剪。默认为 `0`。
-   `axis (int)`: 执行裁剪操作的轴。通常是时间轴。默认为 `-1`（最后一个轴）。
-   `source (str, optional)`: 输入字典中的键，该键包含待转换的信号数据（表现为一个包含 'data'、'channels' 等的信号字典）。默认为 `None`。
-   `target (str, optional)`: 输出字典中的键，转换后（裁剪后）的信号数据将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Crop) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Crop

# 1. 准备一个包含初始信号数据的示例 'results' 字典
results = {
    'raw_signal': {
        'data': np.array([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                           [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]]), # 2 通道, 10 时间点
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 实例化 Crop 变换
#    沿最后一个轴（时间轴）从左侧裁剪 2 个点，从右侧裁剪 3 个点。
cropper = Crop(crop_left=2, crop_right=3, axis=-1, source='raw_signal', target='cropped_signal')

# 3. 应用变换
processed_results = cropper(results)

# 4. 'processed_results' 现在将包含 'cropped_signal' 键。
#    数据将被裁剪:
#    原始数据: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#    crop_left=2 后: [3, 4, 5, 6, 7, 8, 9, 10]
#    再 crop_right=3 (从原始末尾算起) 后: [3, 4, 5, 6, 7]
#    所以, CH1 的数据: [3, 4, 5, 6, 7]
#    所以, CH2 的数据: [13, 14, 15, 16, 17]
#    形状将是 (2, 5)。
# processed_results['cropped_signal'] 的示例内容:
# {
#   'data': np.array([[ 3,  4,  5,  6,  7],
#                      [13, 14, 15, 16, 17]]),
#   'channels': ['CH1', 'CH2'],
#   'freq': 100.0
# }
# print(processed_results['cropped_signal'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Filter

~~~python
class Filter(BaseTransform):
    def __init__(self, l_freq=None, h_freq=None, filter_length="auto", l_trans_bandwidth="auto", h_trans_bandwidth="auto", method="fir", iir_params=None, phase="zero", fir_window="hamming", fir_design="firwin", pad="reflect_limited", source: Optional[str] = None, target: Optional[str] = None):
~~~

对信号数据应用滤波器，利用 `mne.filter.filter_data`。可以配置为带通、低通或高通滤波器。

**参数**

-   `l_freq (float | None)`: 低通带边缘。如果为 `None`，则作为低通滤波器。
-   `h_freq (float | None)`: 高通带边缘。如果为 `None`，则作为高通滤波器。
-   `filter_length (str | int)`: FIR 滤波器的长度（如果适用）。默认为 `"auto"`。
-   `l_trans_bandwidth (str | float)`: 低过渡带宽度。默认为 `"auto"`。
-   `h_trans_bandwidth (str | float)`: 高过渡带宽度。默认为 `"auto"`。
-   `method (str)`: 滤波方法，`"fir"` 或 `"iir"`。默认为 `"fir"`。
-   `iir_params (dict | None)`: 用于 IIR 滤波的参数字典。默认为 `None`。
-   `phase (str)`: 滤波器相位，`"zero"` 或 `"zero-double"`。默认为 `"zero"`。
-   `fir_window (str)`: FIR 设计中使用的窗口。默认为 `"hamming"`。
-   `fir_design (str)`: FIR 设计方法，例如 `"firwin"`。默认为 `"firwin"`。
-   `pad (str)`: 使用的填充类型。默认为 `"reflect_limited"`。
-   `source (Optional[str])`: 输入信号字典的键。默认为 `None`。
-   `target (Optional[str])`: 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Filter) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Filter

# 1. 准备一个示例 'results' 字典
results = {
    'raw_signal': {
        'data': np.random.randn(2, 1000), # 2 通道, 1000 时间点
        'freq': 250.0, # 采样频率
        'channels': ['CH1', 'CH2']
    }
}

# 2. 实例化一个带通滤波器 (例如, 1 Hz 到 40 Hz)
bandpass_filter = Filter(l_freq=1.0, h_freq=40.0, method="fir", source='raw_signal', target='filtered_signal')

# 3. 应用变换
processed_results = bandpass_filter(results)

# 4. 'processed_results' 现在将包含带有带通滤波后数据的 'filtered_signal'。
#    processed_results['filtered_signal'] 中的 'data' 数组将被修改。
# print(processed_results['filtered_signal']['data'])
~~~

---

[`返回顶部`](#tyeedatasettransform)

## NotchFilter

~~~python
class NotchFilter(BaseTransform):
    def __init__(self, freqs: List[float], filter_length="auto", notch_widths=None, trans_bandwidth=1, method="fir", iir_params=None, mt_bandwidth=None, p_value=0.05, phase="zero", fir_window="hamming", fir_design="firwin", pad="reflect_limited", source: Optional[str] = None, target: Optional[str] = None):
~~~

对信号数据应用陷波滤波器，以去除特定的频率成分（例如，工频噪声），使用 `mne.filter.notch_filter`。

**参数**

-   `freqs (List[float])`: 要滤除的频率列表。
-   `filter_length (str | int)`: FIR 滤波器的长度。默认为 `"auto"`。
-   `notch_widths (float | List[float] | None)`: 每个频率的陷波宽度。如果为 `None`，则设置为 `freqs / 200`。
-   `trans_bandwidth (float)`: 过渡带宽度。默认为 `1`。
-   `method (str)`: 滤波方法，例如 `"fir"`, `"iir"`, 或 `"spectrum_fit"`。默认为 `"fir"`。
-   `iir_params (dict | None)`: IIR 滤波参数。默认为 `None`。
-   `mt_bandwidth (float | None)`: 多窗谱带宽 (如果 `method='spectrum_fit'`)。
-   `p_value (float)`: 检测显著峰值的P值 (如果 `method='spectrum_fit'`)。
-   `phase (str)`: 滤波器相位。默认为 `"zero"`。
-   `fir_window (str)`: FIR 设计窗口。默认为 `"hamming"`。
-   `fir_design (str)`: FIR 设计方法。默认为 `"firwin"`。
-   `pad (str)`: 填充类型。默认为 `"reflect_limited"`。
-   `source (Optional[str])`: 输入信号字典的键。默认为 `None`。
-   `target (Optional[str])`: 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, NotchFilter) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import NotchFilter

# 1. 准备一个示例 'results' 字典
results = {
    'noisy_signal': {
        'data': np.random.randn(2, 1000), # 2 通道, 1000 时间点
        'freq': 250.0, # 采样频率
        'channels': ['CH1', 'CH2']
    }
}

# 2. 实例化一个陷波滤波器 (例如, 去除 50 Hz 和 100 Hz 噪声)
powerline_filter = NotchFilter(freqs=[50.0, 100.0], source='noisy_signal', target='denoised_signal')

# 3. 应用变换
processed_results = powerline_filter(results)

# 4. 'processed_results' 现在将包含 'denoised_signal'，其中指定的频率已被衰减。
# print(processed_results['denoised_signal']['data'])
~~~

---

[`返回顶部`](#tyeedatasettransform)

## Cheby2Filter

~~~python
class Cheby2Filter(BaseTransform):
    def __init__(self, l_freq, h_freq, order=6, rp=0.1, rs=60, btype='bandpass', source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `scipy.signal.cheby2` 和 `scipy.signal.filtfilt`（用于零相位滤波）对信号数据应用切比雪夫II型数字或模拟滤波器。

**参数**

-   `l_freq (float)`: 低截止频率。
-   `h_freq (float)`: 高截止频率。
-   `order (int)`: 滤波器阶数。默认为 `6`。
-   `rp (float)`: 对于切比雪夫I型滤波器，通带中单位增益以下允许的最大波纹。Cheby2不使用此参数。默认为 `0.1`。
-   `rs (float)`: 对于切比雪夫II型滤波器，阻带中要求的最小衰减。默认为 `60` (dB)。
-   `btype (str)`: 滤波器类型。选项包括 `'bandpass'`, `'lowpass'`, `'highpass'`, `'bandstop'`。默认为 `'bandpass'`。
-   `source (Optional[str])`: 输入信号字典的键。默认为 `None`。
-   `target (Optional[str])`: 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Cheby2Filter) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Cheby2Filter

# 1. 准备一个示例 'results' 字典
results = {
    'raw_data': {
        'data': np.random.randn(3, 1280), # 3 通道, 1280 时间点
        'freq': 500.0, # 采样频率
        'channels': ['EEG1', 'EEG2', 'EEG3']
    }
}

# 2. 实例化一个 Cheby2Filter (例如, 0.5 Hz 到 45 Hz 的带通)
cheby_filter = Cheby2Filter(l_freq=0.5, h_freq=45.0, order=5, rs=50, source='raw_data', target='cheby_filtered_data')

# 3. 应用变换
processed_results = cheby_filter(results)

# 4. 'processed_results' 现在将包含带有滤波后信号的 'cheby_filtered_data'。
# print(processed_results['cheby_filtered_data']['data'])
~~~

---

[`返回顶部`](#tyeedatasettransform)

## Detrend

~~~python
class Detrend(BaseTransform):
    def __init__(self, axis=-1, source=None, target=None):
~~~

沿指定轴移除信号数据中的线性趋势，使用 `scipy.signal.detrend`。

**参数**

-   `axis (int)`: 移除趋势的数据轴。默认为 `-1`（最后一个轴）。
-   `source (str, optional)`: 输入信号字典的键。默认为 `None`。
-   `target (str, optional)`: 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Detrend) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Detrend

# 1. 创建一个带线性趋势的信号
sfreq = 100.0
time = np.arange(0, 10, 1/sfreq) # 10 秒数据
trend = 0.5 * time # 线性趋势
signal_component = np.sin(2 * np.pi * 5 * time) # 5 Hz 正弦波
original_data_ch1 = signal_component + trend
original_data_ch2 = np.cos(2 * np.pi * 10 * time) + 0.3 * time # 另一个带趋势的通道

results = {
    'trended_signal': {
        'data': np.array([original_data_ch1, original_data_ch2]),
        'freq': sfreq,
        'channels': ['CH1_trend', 'CH2_trend']
    }
}

# 2. 实例化 Detrend 变换
detrender = Detrend(axis=-1, source='trended_signal', target='detrended_signal')

# 3. 应用变换
processed_results = detrender(results)

# 4. 'processed_results' 现在将包含 'detrended_signal'，其中线性趋势已被移除。
# print(processed_results['detrended_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ForEach

~~~python
class ForEach(BaseTransform):
    def __init__(self, transforms: List[BaseTransform], source=None, target=None):
~~~

沿第一个维度，对每个片段（segment）分别应用一系列变换。适用于2D或3D数据（例如，shape=(N, ...)，N为片段数、试验(epoch)数或通道数）。在 `transforms` 列表中提供的子变换在初始化时必须不设置 `source` 或 `target` 参数，因为它们直接操作内部传递的单个片段。

**参数**

-   `transforms (List[BaseTransform])`: 一个变换实例的列表，这些变换将按顺序应用于每个片段。此列表中的每个变换在初始化时都不应设置 `source` 或 `target`。
-   `source (str, optional)`: 输入字典中的键，该键包含待处理的信号数据（表现为一个包含 'data'、'channels' 等的信号字典）。此键下的 'data' 应至少为2D。默认为 `None`。
-   `target (str, optional)`: 输出字典中的键，处理后的信号数据（每个片段都经过转换）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Crop, Detrend, ForEach) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Crop, Detrend, ForEach

# 1. 准备一个包含3D数据（例如，试验 x 通道 x 时间点）的示例 'results' 字典
results = {
    'epoched_signal': {
        'data': np.random.randn(2, 3, 100), # 2 个试验, 每个试验3个通道, 100个时间点
        'channels': ['CH1', 'CH2', 'CH3'], # 通道信息在各试验间保持一致
        'freq': 100.0
    }
}

# 2. 实例化将应用于每个试验的子变换。
#    这些变换在初始化时没有 source/target。
crop_each_epoch = Crop(crop_left=10, crop_right=10, axis=-1) # 裁剪每个试验的时间轴
detrend_each_epoch = Detrend(axis=-1) # 对每个试验的时间轴去趋势

# 3. 实例化 ForEach 变换。
#    它将遍历 'epoched_signal']['data'] 的第一个维度（即试验）。
process_each_epoch = ForEach(
    transforms=[
        crop_each_epoch,
        detrend_each_epoch
    ],
    source='epoched_signal',
    target='processed_epochs'
)

# 4. 应用 ForEach 变换
processed_results = process_each_epoch(results)

# 5. 'processed_results' 现在将包含 'processed_epochs'。
#    'processed_epochs' 中的 'data' 的形状将是 (2, 3, 80)，因为：
#    - ForEach 迭代 2 次（每个试验一次）。
#    - 每个试验 (3x100) 首先被 crop_each_epoch 裁剪为 (3x80)。
#    - 然后，每个裁剪后的试验 (3x80) 被 detrend_each_epoch 去趋势。
#    - 最终得到的 (3x80) 的片段被重新堆叠起来，结果是 (2, 3, 80)。
#    'channels' 和 'freq' 通常会从输入信号字典中保留下来。
#
# processed_results['processed_epochs'] 的示例内容:
# {
#   'data': np.ndarray 类型，形状为 (2, 3, 80), # 2个试验, 3个通道, 80个时间点的数据
#   'channels': ['CH1', 'CH2', 'CH3'],
#   'freq': 100.0
# }
# print(processed_results['processed_epochs']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## Lambda

~~~python
class Lambda(BaseTransform):
    def __init__(self, lambd: Callable, source: Optional[str] = None, target: Optional[str] = None):
~~~

对输入信号字典中的 'data' 字段应用用户定义的 lambda 函数或任何可调用对象。这允许对信号数据进行灵活的自定义操作。

**参数**

-   `lambd (Callable)`: 一个可调用对象（例如 lambda 函数或常规函数），它接收信号数据（例如 NumPy 数组）作为输入，并返回转换后的数据。
-   `source (str, optional)`: 输入字典中的键，该键包含待转换的信号字典（包含 'data'、'channels' 等）。默认为 `None`。
-   `target (str, optional)`: 输出字典中的键，转换后的信号字典将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Lambda) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from typing import Callable, Optional # 用于 Callable, Optional 类型提示
# from tyee.dataset.transform import Lambda

# 1. 准备一个示例 'results' 字典
results = {
    'raw_signal': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 2 通道, 3 时间点
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 定义一个用于自定义操作的 lambda 函数 (例如，将数据乘以 10)
multiply_by_ten = lambda x: x * 10

# 3. 实例化 Lambda 变换
custom_transform = Lambda(
    lambd=multiply_by_ten,
    source='raw_signal',
    target='transformed_signal'
)

# 4. 应用变换
processed_results = custom_transform(results)

# 5. 'processed_results' 现在将包含 'transformed_signal'。
#    'transformed_signal' 中的 'data' 将是原始数据乘以 10 的结果。
# processed_results['transformed_signal'] 的示例内容:
# {
#   'data': np.array([[10, 20, 30], [40, 50, 60]]),
#   'channels': ['CH1', 'CH2'],
#   'freq': 100.0
# }
# print(processed_results['transformed_signal'])

# 使用不同 lambda 的示例，例如，选择第一个通道的数据
select_first_channel_data = lambda x: x[0:1, :] # 保持其为二维数组
channel_selector = Lambda(lambd=select_first_channel_data, source='raw_signal', target='first_channel_data')
selected_channel_results = channel_selector(results)
# selected_channel_results['first_channel_data']['data'] 将是 np.array([[1, 2, 3]])
# 注意：此 lambda 仅更改 'data'。信号字典中的 'channels' 列表
# 仍将是 ['CH1', 'CH2']，除非 lambda 或其他变换更新了它。
# print(selected_channel_results['first_channel_data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Mapping

~~~python
class Mapping(BaseTransform):
    def __init__(self, mapping: dict, source: str = None, target: str = None):
~~~

对输入信号字典中 'data' 字段的值应用字典映射。如果 'data' 是标量，则直接映射。如果是 NumPy 数组或其他可迭代对象，则每个元素都会被单独映射，对于 NumPy 数组会保留其原始形状。

**参数**

-   `mapping (dict)`: 一个字典，其中键是原始值，值是它们应映射到的目标值。
-   `source (str, optional)`: 输入字典中的键，该键包含待转换的信号字典（包含 'data' 等）。默认为 `None`。
-   `target (str, optional)`: 输出字典中的键，转换后的信号字典（包含已映射的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Mapping) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Mapping

# 1. 准备一个示例 'results' 字典
results_scalar = {
    'label_data': {
        'data': 1, # 标量数据
        'channels': ['EventLabel'], # 概念上的通道
    }
}
results_array = {
    'categorical_data': {
        'data': np.array([[0, 1, 2], [2, 0, 1]]), # 数组数据
        'channels': ['FeatureSet1', 'FeatureSet2'], # 概念上的通道
    }
}

# 2. 定义一个映射字典
category_mapping = {
    0: 100, # 将 0 映射到 100
    1: 200, # 将 1 映射到 200
    2: 300  # 将 2 映射到 300
}

# 3. 实例化用于标量数据的 Mapping 变换
map_transform_scalar = Mapping(
    mapping=category_mapping,
    source='label_data',
    target='mapped_label'
)

# 4. 实例化用于数组数据的 Mapping 变换
map_transform_array = Mapping(
    mapping=category_mapping,
    source='categorical_data',
    target='mapped_array'
)

# 5. 对标量数据应用变换
processed_results_scalar = map_transform_scalar(results_scalar)
# processed_results_scalar['mapped_label']['data'] 将是 200 (因为原始数据是 1)

# 6. 对数组数据应用变换
processed_results_array = map_transform_array(results_array)
# processed_results_array['mapped_array']['data'] 将是:
# np.array([[100, 200, 300], [300, 100, 200]])

# print("映射后的标量数据:", processed_results_scalar['mapped_label']['data'])
# print("映射后的数组数据:\n", processed_results_array['mapped_array']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Concat

~~~python
class Concat(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
~~~

沿指定轴连接来自输入信号字典列表中的 'data' 数组。如果 `source` 参数是键的列表，则变换期望接收相应的信号字典列表。它会尝试合并 'channels' 列表，并如果可用，则保留第一个信号的 'freq'。如果 `source` 是单个字符串，则暗示输入信号字典本身可能包含适合某种形式连接的数据（尽管所提供的 `transform` 代码主要处理用于连接的源列表情况）。

**参数**

-   `axis (int)`: 将沿其连接 'data' 数组的轴。默认为 `0`。
-   `source (Optional[Union[str, List[str]]])`: 标识主结果字典中输入信号字典的字符串键列表，或单个字符串键。如果是列表，则处理相应的信号字典。默认为 `None`。
-   `target (Optional[str])`: 输出字典中的键，新的信号字典（包含连接后的 'data'、合并的 'channels' 和 'freq'）将存储在此处。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Concat) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Concat

# 1. 准备一个包含多个信号源的示例 'results' 字典
results = {
    'signal_A': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 形状 (2, 3)
        'channels': ['A_CH1', 'A_CH2'],
        'freq': 100.0
    },
    'signal_B': {
        'data': np.array([[7, 8, 9]]), # 形状 (1, 3)
        'channels': ['B_CH1'],
        'freq': 100.0 # 假设频率相同
    },
    'signal_C': {
        'data': np.array([[10, 11, 12], [13, 14, 15]]), # 形状 (2,3)
        'channels': 'C_CH_Group', # 非列表通道的示例
        'freq': 100.0
    }
}

# 2. 实例化 Concat 变换，沿轴 0 (通道) 连接
#    'source' 是键的列表。
concatenator = Concat(
    axis=0,
    source=['signal_A', 'signal_B', 'signal_C'],
    target='concatenated_signal'
)

# 3. 应用变换
#    BaseTransform 的 __call__ 方法应处理从源键收集信号。
processed_results = concatenator(results)

# 4. 'processed_results' 现在将包含 'concatenated_signal'。
#    'data' 将是 signal_A['data'] 和 signal_B['data'] 沿轴 0 连接的结果。
#    形状: (2+1+2, 3) = (5, 3)
#    'channels' 将是 ['A_CH1', 'A_CH2', 'B_CH1', 'C_CH_Group']
#    'freq' 将是 100.0
#
# processed_results['concatenated_signal'] 的示例内容:
# {
#   'data': np.array([[ 1,  2,  3],
#                      [ 4,  5,  6],
#                      [ 7,  8,  9],
#                      [10, 11, 12],
#                      [13, 14, 15]]),
#   'channels': ['A_CH1', 'A_CH2', 'B_CH1', 'C_CH_Group'],
#   'freq': 100.0
# }
# print(processed_results['concatenated_signal'])
~~~

---

[`返回顶部`](#tyeedatasettransform)

## Stack

~~~python
class Stack(BaseTransform):
    def __init__(self, axis: int = 0, source: Optional[Union[str, List[str]]] = None, target: Optional[str] = None):
~~~

沿新轴堆叠来自输入信号字典列表中的 'data' 数组。如果 `source` 参数是键的列表，则变换期望接收相应的信号字典列表。这些字典中的 'data' 数组必须具有兼容的形状才能进行堆叠。如果 `source` 是单个字符串，则暗示输入信号字典本身可能被处理（尽管所提供的 `transform` 代码主要处理用于堆叠的源列表情况）。

**参数**

-   `axis (int)`: 将沿其堆叠 'data' 数组的轴。这将是结果 'data' 数组中的一个新轴。默认为 `0`。
-   `source (Optional[Union[str, List[str]]])`: 标识主结果字典中输入信号字典的字符串键列表，或单个字符串键。如果是列表，则处理相应的信号字典。默认为 `None`。
-   `target (Optional[str])`: 输出字典中的键，新的信号字典（包含堆叠后的 'data'）将存储在此处。此变换的当前实现未明确合并或保留像 'channels' 和 'freq' 这样的元数据，仅在新字典中返回 'data'。默认为 `None`。

**使用样例**

~~~python
# 假设必要的类 (BaseTransform, Stack) 和 numpy (as np) 已导入。
# 例如:
# import numpy as np
# from tyee.dataset.transform import Stack

# 1. 准备一个包含多个信号源的示例 'results' 字典
#    用于堆叠的 Data 数组应具有相同的形状。
results = {
    'epoch_1': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 形状 (2 通道, 3 时间点)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    },
    'epoch_2': {
        'data': np.array([[7, 8, 9], [10, 11, 12]]), # 形状 (2 通道, 3 时间点)
        'channels': ['CH1', 'CH2'], # 为简单起见，假设通道相同
        'freq': 100.0
    },
    'epoch_3': {
        'data': np.array([[13,14,15], [16,17,18]]), # 形状 (2 通道, 3 时间点)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 实例化 Stack 变换，沿新的第一个轴 (axis=0) 堆叠
#    这将为试验(epochs)创建一个新维度。
stacker = Stack(
    axis=0,
    source=['epoch_1', 'epoch_2', 'epoch_3'],
    target='stacked_epochs'
)

# 3. 应用变换
processed_results = stacker(results)

# 4. 'processed_results' 现在将包含 'stacked_epochs'。
#    'data' 将是 epoch_1['data'], epoch_2['data'], 和 epoch_3['data'] 堆叠的结果。
#    每个数据的原始形状: (2, 3)
#    沿 axis=0 堆叠后的新形状: (3 个试验, 2 个通道, 3 个时间点)
#
# processed_results['stacked_epochs'] 的示例内容:
# {
#   'data': np.array([[[ 1,  2,  3], [ 4,  5,  6]],
#                      [[ 7,  8,  9], [10, 11, 12]],
#                      [[13, 14, 15], [16, 17, 18]]])
# }
# 注意: 当前 Stack 变换的实现仅返回 {'data': stacked_data}。
# 它不传播 'channels' 或 'freq'。
# print(processed_results['stacked_epochs']['data'].shape)
# print(processed_results['stacked_epochs'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ZScoreNormalize

~~~python
class ZScoreNormalize(BaseTransform):
    def __init__(
        self, 
        mean: Optional[np.ndarray] = None, 
        std: Optional[np.ndarray] = None, 
        axis: Optional[int] = None, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

对信号字典的 'data' 字段执行 Z-score 标准化。可以沿指定轴进行，并可选择使用用户提供的均值和标准差。如果提供的自定义均值/标准差是一维数组且指定了轴，它将被重塑以适配广播运算。如果标准差为零，则将其视为1以避免除零错误。

**参数**

- **mean** (`Optional[np.ndarray]`): 用于标准化的预计算均值。如果为 `None`，则从数据中计算均值。默认为 `None`。
- **std** (`Optional[np.ndarray]`): 预计算的标准差。如果为 `None`，则从数据中计算标准差。默认为 `None`。
- **axis** (`Optional[int]`): 如果未提供均值和标准差，则沿此轴计算它们。如果为 `None`，则对整个数据进行标准化。默认为 `None`。
- **epsilon** (`float`): 添加到标准差的一个小值，以防止除以零。默认为 `1e-8`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 ZScoreNormalize 已导入, numpy as np
# from tyee.dataset.transform import ZScoreNormalize
# import numpy as np

# 1. 准备示例数据
results = {
    'raw_signal': {
        'data': np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 实例化 ZScoreNormalize 以沿行标准化 (axis=1)
normalizer_axis1 = ZScoreNormalize(axis=1, source='raw_signal', target='normalized_axis1')
processed_results_axis1 = normalizer_axis1(results)

# 3. 实例化 ZScoreNormalize 进行整体标准化 (axis=None)
normalizer_overall = ZScoreNormalize(axis=None, source='raw_signal', target='normalized_overall')
processed_results_overall = normalizer_overall(results.copy()) # 为独立处理使用副本

# print("沿轴 1 标准化:\n", processed_results_axis1['normalized_axis1']['data'])
# print("整体标准化:\n", processed_results_overall['normalized_overall']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## MinMaxNormalize

~~~python
class MinMaxNormalize(BaseTransform):
    def __init__(
        self, 
        min: Optional[Union[np.ndarray, float]] = None, 
        max: Optional[Union[np.ndarray, float]] = None, 
        axis: Optional[int] = None, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

对信号字典的 'data' 字段执行最小-最大标准化，将数据缩放到一个范围（通常是0到1）。它支持沿指定轴进行标准化，并可以使用自定义的最小值和最大值。如果提供的自定义最小值/最大值是一维数组且指定了轴，它们将被扩展维度以适配数据的广播运算。

**参数**

- **min** (`Optional[Union[np.ndarray, float]]`): 预计算的最小值或数组。如果为 `None`，则从数据中计算最小值。默认为 `None`。
- **max** (`Optional[Union[np.ndarray, float]]`): 预计算的最大值或数组。如果为 `None`，则从数据中计算最大值。默认为 `None`。
- **axis** (`Optional[int]`): 如果未提供最小值和最大值，则沿此轴计算它们。如果为 `None`，则对整个数据进行标准化。默认为 `None`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 MinMaxNormalize 已导入, numpy as np
# from tyee.dataset.transform import MinMaxNormalize
# import numpy as np

# 1. 准备示例数据
results = {
    'raw_data': {
        'data': np.array([[10., 20., 30.], [0., 50., 100.]]),
        'channels': ['A', 'B'],
        'freq': 100.0
    }
}

# 2. 实例化 MinMaxNormalize 以沿列标准化 (axis=0)
minmax_axis0 = MinMaxNormalize(axis=0, source='raw_data', target='minmax_axis0')
processed_axis0 = minmax_axis0(results)

# 3. 实例化 MinMaxNormalize 进行整体标准化
minmax_overall = MinMaxNormalize(source='raw_data', target='minmax_overall')
processed_overall = minmax_overall(results.copy())

# print("沿轴 0 标准化:\n", processed_axis0['minmax_axis0']['data'])
# print("整体标准化:\n", processed_overall['minmax_overall']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## QuantileNormalize

~~~python
class QuantileNormalize(BaseTransform):
    def __init__(
        self, 
        q: float = 0.95, 
        axis: Optional[int] = -1, 
        epsilon: float = 1e-8, 
        source: Optional[str] = None, 
        target: Optional[str] = None
    ):
~~~

对 'data' 字段执行分位数标准化。该方法通过将数据除以其绝对值的指定分位数来进行标准化。它可以沿特定轴应用。

**参数**

- **q** (`float`): 要计算的分位数（介于0和1之间）。默认为 `0.95`。
- **axis** (`Optional[int]`): 沿其计算分位数的轴。默认为 `-1`（最后一个轴）。
- **epsilon** (`float`): 添加到分位数值的一个小值，以防止除以零。默认为 `1e-8`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 QuantileNormalize 已导入, numpy as np
# from tyee.dataset.transform import QuantileNormalize
# import numpy as np

# 1. 准备示例数据
results = {
    'signal': {
        'data': np.array([[-10., 0., 10., 20., 100.], [1., 2., 3., 4., 5.]]),
        'channels': ['S1', 'S2'],
        'freq': 100.0
    }
}

# 2. 实例化 QuantileNormalize，使用第90百分位数并沿行标准化 (axis=1)
quantile_norm = QuantileNormalize(q=0.9, axis=1, source='signal', target='q_normalized_signal')
processed_results = quantile_norm(results)

# print("分位数标准化信号 (q=0.9, axis=1):\n", processed_results['q_normalized_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## RobustNormalize

~~~python
class RobustNormalize(BaseTransform):
    def __init__(
        self,
        median: Optional[np.ndarray] = None,
        iqr: Optional[np.ndarray] = None,
        quantile_range: tuple = (25.0, 75.0),
        axis: Optional[int] = None,
        epsilon: float = 1e-8,
        unit_variance: bool = False,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

执行稳健标准化，公式为 `(x - median) / IQR`，其中 IQR (四分位距) 是 `q_max分位 - q_min分位`，此方法对异常值具有抵抗性. 它支持沿指定轴进行标准化，并可选择将结果缩放到单位方差. 也可以提供自定义的中位数和IQR值.

**参数**

- **median** (`Optional[np.ndarray]`): 预计算的中位数. 如果为 `None`，则从数据中计算. 默认为 `None`.
- **iqr** (`Optional[np.ndarray]`): 预计算的四分位距. 如果为 `None`，则使用 `quantile_range` 从数据中计算. 默认为 `None`.
- **quantile_range** (`tuple`): 两个浮点数的元组 `(q_min, q_max)`，表示用于IQR计算的较低和较高百分位数（0-100）. 默认为 `(25.0, 75.0)`.
- **axis** (`Optional[int]`): 沿其计算中位数和IQR的轴. 如果为 `None`，则对整个数据进行计算. 默认为 `None`.
- **epsilon** (`float`): 添加到IQR的一个小值，以防止除以零. 默认为 `1e-8`.
- **unit_variance** (`bool`): 如果为 `True`，则缩放输出，使得IQR对应于标准正态分布的 `norm.ppf(q_min/100.0)` 和 `norm.ppf(q_max/100.0)` 之间的范围，从而有效地使数据中心部分的方差接近1. 默认为 `False`.
- **source** (`Optional[str]`): 输入信号字典的键. 默认为 `None`.
- **target** (`Optional[str]`): 输出信号字典的键. 默认为 `None`.

**使用样例**

~~~python
# 假设 BaseTransform 和 RobustNormalize 已导入, numpy as np
# from tyee.dataset.transform import RobustNormalize
# import numpy as np

# 1. 准备示例数据，包含一个异常值
results = {
    'measurements': {
        'data': np.array([[1., 2., 3., 4., 5., 100.], [6., 7., 8., 9., 10., -50.]]),
        'channels': ['Sensor1', 'Sensor2']
    }
}

# 2. 实例化 RobustNormalize 以沿行标准化 (axis=1)
robust_scaler_axis1 = RobustNormalize(axis=1, source='measurements', target='robust_scaled_axis1')
processed_axis1 = robust_scaler_axis1(results)

# 3. 实例化 RobustNormalize 进行整体标准化并启用 unit_variance
robust_scaler_overall_uv = RobustNormalize(unit_variance=True, source='measurements', target='robust_scaled_overall_uv')
processed_overall_uv = robust_scaler_overall_uv(results.copy())

# print("稳健缩放 (axis=1):\n", processed_axis1['robust_scaled_axis1']['data'])
# print("稳健缩放 (整体, 单位方差):\n", processed_overall_uv['robust_scaled_overall_uv']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Baseline

~~~python
class Baseline(BaseTransform):
    def __init__(
        self,
        baseline_start: Optional[int] = None,
        baseline_end: Optional[int] = None,
        axis: Optional[int] = -1,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

对信号数据进行基线校正：减去指定区间的均值。支持自定义基线区间（例如前N个采样点或任意区间）。`axis` 参数控制基线均值的计算方式：`-1`（默认）表示每个通道独立进行基线校正，`None` 表示在基线周期内的所有通道和所有采样点使用同一个基线值。

**参数**

- **baseline_start** (`Optional[int]`): 基线周期的起始采样点索引。如果为 `None`，则从头开始（索引0）。默认为 `None`。
- **baseline_end** (`Optional[int]`): 基线周期的终止采样点索引（不包含此点）。如果为 `None`，则延伸到信号末尾。默认为 `None`。
- **axis** (`Optional[int]`): 计算基线均值所沿的轴。如果为 `-1`，则在基线周期内为每个通道独立计算时间样本的均值。如果为 `None`，则在基线周期内计算所有通道和所有时间样本的单个均值。默认为 `-1`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Baseline 已导入, numpy as np
# from tyee.dataset.transform import Baseline
# import numpy as np

# 1. 准备示例数据
data_array = np.array([[1., 2., 10., 11., 12.], [20., 21., 30., 31., 32.]]) # 2 通道, 5 时间点
results = {
    'trial_data': {
        'data': data_array,
        'channels': ['ChA', 'ChB'],
        'freq': 100.0
    }
}

# 2. 实例化 Baseline，使用前2个采样点对每个通道进行基线校正
baseline_corrector = Baseline(baseline_start=0, baseline_end=2, axis=-1, source='trial_data', target='baselined_data')
processed_results = baseline_corrector(results)

# ChA 的基线: mean([1,2]) = 1.5。校正后的 ChA: [-0.5, 0.5, 8.5, 9.5, 10.5]
# ChB 的基线: mean([20,21]) = 20.5。校正后的 ChB: [-0.5, 0.5, 9.5, 10.5, 11.5]
# print("基线校正后的数据 (每通道):\n", processed_results['baselined_data']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Mean

~~~python
class Mean(BaseTransform):
    def __init__(self, axis: Optional[Union[int, tuple]] = None, source: Optional[str] = None, target: Optional[str] = None, keepdims: bool = False):
~~~

计算信号字典中 'data' 字段沿指定轴或多个轴的均值。原始信号字典的结构被保留，但其 'data' 字段被计算得到的均值替换。

**参数**

- **axis** (`Optional[Union[int, tuple]]`): 计算均值所沿的一个或多个轴。默认情况是计算扁平化数组的均值。默认为 `None`。
- **keepdims** (`bool`): 如果设置为 `True`，则被规约的轴将作为尺寸为1的维度保留在结果中。通过此选项，结果将能正确地与输入数组进行广播。默认为 `False`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Mean 已导入, numpy as np
# from tyee.dataset.transform import Mean
# import numpy as np

# 1. 准备示例数据
results = {
    'time_series_data': {
        'data': np.array([[[1., 2., 3.], [4., 5., 6.]], [[7., 8., 9.], [10., 11., 12.]]]), # 形状 (2, 2, 3)
        'channels': ['Epoch1_CH1', 'Epoch1_CH2', 'Epoch2_CH1', 'Epoch2_CH2'], # 概念上的通道
        'freq': 100.0
    }
}

# 2. 实例化 Mean 以计算沿最后一个轴 (axis=-1) 的均值，并保持维度 (keepdims=True)
mean_transform = Mean(axis=-1, keepdims=True, source='time_series_data', target='mean_over_time')
processed_results = mean_transform(results)

# 原始数据形状 (2,2,3)
# 沿 axis=-1 计算均值并 keepdims=True 会得到形状 (2,2,1)
# 第一个epoch，第一个通道的数据: mean([1,2,3]) = 2.0
# 第一个epoch，第二个通道的数据: mean([4,5,6]) = 5.0
# print("按时间计算的均值 (keepdims=True):\n", processed_results['mean_over_time']['data'])
# print("均值变换后的形状:", processed_results['mean_over_time']['data'].shape)

# 整体均值的示例
overall_mean_transform = Mean(axis=None, source='time_series_data', target='overall_mean')
overall_mean_results = overall_mean_transform(results.copy())
# print("整体均值:", overall_mean_results['overall_mean']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## OneHotEncode

~~~python
class OneHotEncode(BaseTransform):
    def __init__(self, num: int, source=None, target=None):
~~~

将信号字典中 'data' 字段的输入数据转换为独热编码格式。输入数据可以是标量整数或类似数组的对象（列表、NumPy数组），表示分类标签。`num` 参数指定了总的类别数量，它决定了独热向量的长度。

**参数**

- **num** (`int`): 不同类别的总数。这将是独热编码向量的维度（例如，如果 `num=5`，标签 `2` 将变为 `[0,0,1,0,0]`）。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待进行独热编码的信号字典（包含'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含独热编码后的'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 OneHotEncode 已导入, numpy as np
# from tyee.dataset.transform import OneHotEncode
# import numpy as np

# 1. 准备示例数据 (标量和数组)
results_scalar = {
    'label_scalar': {
        'data': 2, # 标量标签
        'info': '样本标签 A'
    }
}
results_array = {
    'label_array': {
        'data': np.array([0, 1, 3, 1]), # 标签数组
        'info': '样本标签 B'
    }
}

# 2. 定义用于独热编码的总类别数
num_classes = 4

# 3. 实例化用于标量数据的 OneHotEncode
one_hot_encoder_scalar = OneHotEncode(num=num_classes, source='label_scalar', target='encoded_label_scalar')

# 4. 实例化用于数组数据的 OneHotEncode
one_hot_encoder_array = OneHotEncode(num=num_classes, source='label_array', target='encoded_label_array')

# 5. 对标量数据应用变换
processed_scalar = one_hot_encoder_scalar(results_scalar)
# processed_scalar['encoded_label_scalar']['data'] 将是 np.array([0., 0., 1., 0.])

# 6. 对数组数据应用变换
processed_array = one_hot_encoder_array(results_array)
# processed_array['encoded_label_array']['data'] 将是:
# np.array([[1., 0., 0., 0.],
#           [0., 1., 0., 0.],
#           [0., 0., 0., 1.],
#           [0., 1., 0., 0.]])

# print("独热编码后的标量:\n", processed_scalar['encoded_label_scalar']['data'])
# print("独热编码后的数组:\n", processed_array['encoded_label_array']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Pad

~~~python
class Pad(BaseTransform):
    def __init__(
        self,
        pad_len: int,
        axis: int = 0,
        side: str = 'post',
        mode: str = 'constant',
        constant_values: float = 0,
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

对信号字典中的 'data' 数组沿指定轴进行填充。此变换允许在选定轴之前、之后或两侧进行填充，并可使用 `numpy.pad` 支持的各种填充模式。

**参数**

- **pad_len** (`int`): 要填充的元素数量。如果 `side` 是 'both'，则此数量的元素将添加到轴的两侧。
- **axis** (`int`): 应应用填充的轴。默认为 `0`。
- **side** (`str`): 指定是在轴的 'pre'（之前）、'post'（之后）还是 'both'（两侧）进行填充。必须是 'pre'、'post' 或 'both' 之一。默认为 `'post'`。
- **mode** (`str`): 要使用的填充模式，由 `numpy.pad` 定义（例如 'constant', 'reflect', 'edge'）。默认为 `'constant'`。
- **constant_values** (`float`): 当 `mode` 为 'constant' 时用于填充的值。默认为 `0`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待填充的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已填充的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Pad 已导入, numpy as np
# from tyee.dataset.transform import Pad
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'short_signal': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 形状 (2, 3)
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 实例化 Pad 变换，沿轴 1 (时间) 在数据后添加 2 个零
padder_post = Pad(
    pad_len=2,
    axis=1,
    side='post',
    mode='constant',
    constant_values=0,
    source='short_signal',
    target='padded_signal_post'
)
processed_post = padder_post(results)
# processed_post['padded_signal_post']['data'] 将是:
# np.array([[1, 2, 3, 0, 0], [4, 5, 6, 0, 0]])

# 3. 实例化 Pad 变换，沿轴 0 (通道) 在数据前添加 1 个元素 (例如，边缘值)
padder_pre_edge = Pad(
    pad_len=1,
    axis=0,
    side='pre',
    mode='edge', # 边缘填充使用边缘值
    source='short_signal',
    target='padded_signal_pre_edge'
)
processed_pre_edge = padder_pre_edge(results.copy()) # 为独立处理使用副本
# 如果原始数据是 [[1,2,3],[4,5,6]]
# processed_pre_edge['padded_signal_pre_edge']['data'] 将是:
# np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6]])

# print("后填充 (轴=1):\n", processed_post['padded_signal_post']['data'])
# print("前填充并使用边缘值 (轴=0):\n", processed_pre_edge['padded_signal_pre_edge']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Resample

~~~python
class Resample(BaseTransform):
    def __init__(self,
                 desired_freq: Optional[int] = None,
                 axis: int = -1,
                 window: str = "auto",
                 n_jobs: Optional[int] = None,
                 pad: str = "auto",
                 npad: str = 'auto',
                 method: str = "fft",
                 verbose: Optional[bool] = None,
                 source: Optional[str] = None,
                 target: Optional[str] = None):
~~~

使用 `mne.filter.resample` 将信号字典中的 'data' 重采样到新的 `desired_freq`。此变换适用于2D信号（例如，通道 x 时间）。如果当前频率已经是目标频率，或者 `desired_freq` 为 `None`，则不执行重采样。

**参数**

- **desired_freq** (`Optional[int]`): 目标采样频率。如果为 `None` 或等于当前频率，则信号保持不变。
- **axis** (`int`): 沿其重采样数据的轴（通常是时间轴）。默认为 `-1`。
- **window** (`str`): 重采样中使用的窗口。有关选项，请参见 `mne.filter.resample`。默认为 `"auto"`。
- **n_jobs** (`Optional[int]`): 并行运行以进行重采样的作业数。默认为 `None`（通常表示1个作业）。如果未指定为 `None` 以用于并行后端，`mne.filter.resample` 默认使用 `n_jobs=1`。
- **pad** (`str`): 要使用的填充类型。请参见 `mne.filter.resample`。默认为 `"auto"`。
- **npad** (`str`): 应用的填充量。（注意：`mne.filter.resample` 通常使用 `npad='auto'` 或整数。MNE中的参数名称可能略有不同，例如 `n_pad_type` 或与填充长度相关）。默认为 `'auto'`。
- **method** (`str`): 重采样方法。可以是 `'fft'`（基于FFT的重采样）或 `'polyphase'`（多相滤波，通常对于下采样更快）。`mne.filter.resample` 使用此参数。默认为 `"fft"`。
- **verbose** (`Optional[bool]`): 控制MNE重采样函数的详细程度。默认为 `None`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, Resample 已导入, numpy as np.
# from tyee.dataset.transform import Resample
# import numpy as np

# 1. 准备示例数据
results = {
    'raw_signal': {
        'data': np.random.randn(5, 1000), # 5 通道, 1000 时间点
        'freq': 200.0, # 原始采样频率
        'channels': [f'CH{i+1}' for i in range(5)]
    }
}

# 2. 实例化 Resample 将频率更改为 100 Hz
resampler = Resample(desired_freq=100, source='raw_signal', target='resampled_signal')

# 3. 应用变换
processed_results = resampler(results)

# 4. 'processed_results' 将包含 'resampled_signal'
#    'data' 将被重采样, 'freq' 将更新为 100.0。
#    新的时间点数将约为 1000 * (100/200) = 500。
# print(f"原始频率: {results['raw_signal']['freq']}, 新频率: {processed_results['resampled_signal']['freq']}")
# print(f"原始形状: {results['raw_signal']['data'].shape}, 新形状: {processed_results['resampled_signal']['data'].shape}")
~~~

[`返回顶部`](#tyeedatasettransform)

## Downsample

~~~python
class Downsample(BaseTransform):
    def __init__(self, desired_freq: int, axis: int = -1, source=None, target=None):
~~~

通过选择每N个样本（其中N为 `cur_freq // desired_freq`）将信号 'data' 从其当前频率 (`cur_freq`) 下采样到 `desired_freq`。信号字典中的 'freq' 字段会相应更新。如果当前频率已经是目标频率或 `cur_freq` 不可用，则信号保持不变。

**参数**

- **desired_freq** (`int`): 下采样后的目标采样频率。对于简单的抽取，必须是当前频率的约数。
- **axis** (`int`): 沿其进行下采样的轴（通常是时间轴）。默认为 `-1`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, Downsample 已导入, numpy as np.
# from tyee.dataset.transform import Downsample
# import numpy as np

# 1. 准备示例数据
results = {
    'high_freq_signal': {
        'data': np.random.randn(3, 1200), # 3 通道, 1200 时间点
        'freq': 300.0, # 原始采样频率
        'channels': ['X', 'Y', 'Z']
    }
}

# 2. 实例化 Downsample 将频率降低到 100 Hz
downsampler = Downsample(desired_freq=100, source='high_freq_signal', target='downsampled_signal')

# 3. 应用变换
processed_results = downsampler(results)

# 4. 'processed_results' 将包含 'downsampled_signal'
#    'data' 将通过选取每第3个样本 (300/100=3) 来下采样。
#    'freq' 将更新为 100.0。
#    新的时间点数: 1200 / 3 = 400。
# print(f"原始频率: {results['high_freq_signal']['freq']}, 新频率: {processed_results['downsampled_signal']['freq']}")
# print(f"原始形状: {results['high_freq_signal']['data'].shape}, 新形状: {processed_results['downsampled_signal']['data'].shape}")
~~~

[`返回顶部`](#tyeedatasettransform)

## Interpolate

~~~python
class Interpolate(BaseTransform):
    def __init__(self, desired_freq: int, axis: int = -1, kind: str = 'linear', source=None, target=None):
~~~

使用 `scipy.interpolate.interp1d` 通过插值将信号 'data' 从其当前频率 (`cur_freq`) 上采样到 `desired_freq`。信号字典中的 'freq' 字段会相应更新。如果当前频率已经是目标频率或 `cur_freq` 不可用，则信号保持不变。当前实现通过在插值前附加最后一个样本来填充数据，这可能是一种特定的边界处理策略。

**参数**

- **desired_freq** (`int`): 插值后的目标采样频率。
- **axis** (`int`): 沿其进行插值的轴（通常是时间轴）。默认为 `-1`。
- **kind** (`str`): 指定插值类型的字符串（例如，‘linear’, ‘nearest’, ‘zero’, ‘slinear’, ‘quadratic’, ‘cubic’, ‘previous’, ‘next’）。默认为 `'linear'`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, Interpolate 已导入, numpy as np, scipy.interpolate.interp1d
# from tyee.dataset.transform import Interpolate
# import numpy as np
# from scipy.interpolate import interp1d # 类运行需要此导入

# 1. 准备示例数据
results = {
    'low_freq_signal': {
        'data': np.array([[1., 2., 3.], [10., 11., 12.]]), # 2 通道, 3 时间点
        'freq': 50.0, # 原始采样频率
        'channels': ['SensorA', 'SensorB']
    }
}

# 2. 实例化 Interpolate 将频率增加到 100 Hz (使用线性插值)
interpolator = Interpolate(desired_freq=100, kind='linear', source='low_freq_signal', target='interpolated_signal')

# 3. 应用变换
processed_results = interpolator(results)

# 4. 'processed_results' 将包含 'interpolated_signal'
#    'data' 将通过因子 2 (100/50=2) 进行上采样。
#    'freq' 将更新为 100.0。
#    时间点数将大约增加一倍 (由于示例中的特定索引方式，可能会减一)。
#    例如, 对于 [1,2,3] -> ratio=2, old_indices=[0,2,4], new_indices=[0,1,2,3]
#    f(0)=1, f(1)=1.5, f(2)=2, f(3)=2.5
# print(f"原始频率: {results['low_freq_signal']['freq']}, 新频率: {processed_results['interpolated_signal']['freq']}")
# print(f"原始形状: {results['low_freq_signal']['data'].shape}, 新形状: {processed_results['interpolated_signal']['data'].shape}")
# print("插值后的数据:\n", processed_results['interpolated_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Reshape

~~~python
class Reshape(BaseTransform):
    def __init__(self, shape: Tuple[int, ...], source: Optional[str] = None, target: Optional[str] = None):
~~~

一个通用的 reshape transform，使用 `numpy.reshape` 将信号字典中的 'data' 数组重塑为指定的目标形状。重塑后元素的总数必须保持不变。

**参数**

- **shape** (`Tuple[int, ...]`): 数据应重塑为的目标形状。其中一个维度可以是-1，在这种情况下，该值将从数组的长度和其余维度推断出来。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待重塑的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已重塑的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Reshape 已导入, numpy as np.
# from tyee.dataset.transform import Reshape
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'flat_data': {
        'data': np.arange(12), # 一个包含12个元素的扁平数组
        'channels': ['MixedData'],
        'freq': None
    }
}

# 2. 实例化 Reshape 变换，将数据重塑为 (3, 4)
reshaper = Reshape(shape=(3, 4), source='flat_data', target='reshaped_data')

# 3. 应用变换
processed_results = reshaper(results)

# 4. 'processed_results' 现在将包含 'reshaped_data'。
#    'data' 将从 (12,) 重塑为 (3, 4)。
# processed_results['reshaped_data'] 的示例内容:
# {
#   'data': np.array([[ 0,  1,  2,  3],
#                      [ 4,  5,  6,  7],
#                      [ 8,  9, 10, 11]]),
#   'channels': ['MixedData'],
#   'freq': None
# }
# print(processed_results['reshaped_data']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Transpose

~~~python
class Transpose(BaseTransform):
    def __init__(self, axes=None, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.transpose`，通过根据 `axes` 参数排列其轴来转置信号字典中的 'data' 数组。如果 `axes` 为 `None`，它会反转轴的顺序。

**参数**

- **axes** (`tuple of ints, optional`): 一个整数元组或列表，是 `[0, 1, ..., N-1]` 的排列，其中 N 是数据轴的数量。返回数组的第 i 个轴将对应于输入的编号为 `axes[i]` 的轴。如果为 `None`（默认值），则轴被反转（例如，对于2D数组，这是标准转置）。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待转置的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已转置的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Transpose 已导入, numpy as np.
# from tyee.dataset.transform import Transpose
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'original_matrix': {
        'data': np.array([[[0, 1], [2, 3]], [[4, 5], [6, 7]]]), # 形状 (2, 2, 2)
        'info': '一个三维数组'
    }
}

# 2. 实例化 Transpose 以交换轴 0 和轴 2
transposer = Transpose(axes=(2, 1, 0), source='original_matrix', target='transposed_matrix')

# 3. 应用变换
processed_results = transposer(results)
# 原始数据形状: (2, 2, 2)
# 使用 axes=(2,1,0) 转置后的数据形状: (2, 2, 2)

# 4. 实例化 Transpose 进行完全反转 (axes=None 时的默认行为)
full_reverser = Transpose(source='original_matrix', target='reversed_axes_matrix')
processed_reversed = full_reverser(results.copy())

# print("转置后 (2,1,0) 的数据:\n", processed_results['transposed_matrix']['data'])
# print("形状:", processed_results['transposed_matrix']['data'].shape)
# print("轴反转后的数据:\n", processed_reversed['reversed_axes_matrix']['data'])
# print("形状:", processed_reversed['reversed_axes_matrix']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## Squeeze

~~~python
class Squeeze(BaseTransform):
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.squeeze` 从信号字典中 'data' 数组的形状中移除单维度条目。

**参数**

- **axis** (`Optional[int]`): 选择要移除的单维度条目的子集。如果选择的轴的形状条目大于1，则会引发错误。如果为 `None`（默认值），则移除所有单维度条目。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待压缩维度的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已压缩维度的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Squeeze 已导入, numpy as np.
# from tyee.dataset.transform import Squeeze
# import numpy as np

# 1. 准备一个包含具有单一维度的示例 'results' 字典
results = {
    'extra_dim_data': {
        'data': np.array([[[1, 2, 3]]]), # 形状 (1, 1, 3)
        'info': '带有单一维度的数据'
    }
}

# 2. 实例化 Squeeze 以移除所有单一维度
squeezer_all = Squeeze(source='extra_dim_data', target='squeezed_all_data')
processed_all = squeezer_all(results)
# processed_all['squeezed_all_data']['data'] 的形状将是 (3,)

# 3. 实例化 Squeeze 以移除特定的单一维度 (例如, axis 0)
squeezer_axis0 = Squeeze(axis=0, source='extra_dim_data', target='squeezed_axis0_data')
processed_axis0 = squeezer_axis0(results.copy())
# processed_axis0['squeezed_axis0_data']['data'] 的形状将是 (1, 3)

# print("压缩所有单一维度后:\n", processed_all['squeezed_all_data']['data'])
# print("形状:", processed_all['squeezed_all_data']['data'].shape)
# print("压缩轴 0 后:\n", processed_axis0['squeezed_axis0_data']['data'])
# print("形状:", processed_axis0['squeezed_axis0_data']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## ExpandDims

~~~python
class ExpandDims(BaseTransform):
    def __init__(self, axis: Optional[int] = None, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.expand_dims` 通过在扩展数组形状的 `axis` 位置插入一个新轴来扩展信号字典中 'data' 数组的形状。

**参数**

- **axis** (`Optional[int]`): 新轴在扩展后的轴中的位置。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待扩展维度的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已扩展维度的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 ExpandDims 已导入, numpy as np.
# from tyee.dataset.transform import ExpandDims
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'array_2d': {
        'data': np.array([[1, 2, 3], [4, 5, 6]]), # 形状 (2, 3)
        'info': '一个二维数组'
    }
}

# 2. 实例化 ExpandDims 以在开头添加一个新轴 (axis=0)
expander_axis0 = ExpandDims(axis=0, source='array_2d', target='expanded_axis0_data')
processed_axis0 = expander_axis0(results)
# processed_axis0['expanded_axis0_data']['data'] 的形状将是 (1, 2, 3)

# 3. 实例化 ExpandDims 以在末尾添加一个新轴 (对于原始2D数组，axis=-1 或 axis=2)
expander_axis_end = ExpandDims(axis=-1, source='array_2d', target='expanded_axis_end_data')
processed_axis_end = expander_axis_end(results.copy())
# processed_axis_end['expanded_axis_end_data']['data'] 的形状将是 (2, 3, 1)

# print("在轴 0 扩展后:\n", processed_axis0['expanded_axis0_data']['data'])
# print("形状:", processed_axis0['expanded_axis0_data']['data'].shape)
# print("在轴 -1 扩展后:\n", processed_axis_end['expanded_axis_end_data']['data'])
# print("形状:", processed_axis_end['expanded_axis_end_data']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## Insert

~~~python
class Insert(BaseTransform):
    def __init__(self, indices: Union[int, List[int], np.ndarray], value: Union[int, float] = 0, axis: int = 1, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.insert`，在信号字典中的 'data' 数组内，沿指定 `axis` 的给定 `indices` 处插入指定的 `value`。

**参数**

- **indices** (`Union[int, List[int], np.ndarray]`): 值插入的目标位置。如果指定了 `axis`，`indices` 必须是整数或整数列表/数组。如果 `axis` 为 `None`，则 `indices` 是一个一维类数组对象，指定应在扁平化的 `data` 数组的何处插入 `value`。
- **value** (`Union[int, float]`): 要插入的值。如果 `value` 的类型与 `data` 的类型不同，`value` 将被转换为 `data` 的类型。如有必要，`value` 会被广播到正确的形状。默认为 `0`。
- **axis** (`int`): 沿其插入 `value` 的轴。如果为 `None`，则在插入前将 `data` 扁平化。默认为 `1`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待插入值的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（在 'data' 中包含已插入值）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Insert 已导入, numpy as np.
# from tyee.dataset.transform import Insert
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'original_array': {
        'data': np.array([[1, 2], [3, 4], [5, 6]]), # 形状 (3, 2)
        'info': '原始数组'
    }
}

# 2. 实例化 Insert, 沿轴 1 (列) 在索引 1 处插入值 -99
inserter_cols = Insert(indices=1, value=-99, axis=1, source='original_array', target='inserted_cols_array')
processed_cols = inserter_cols(results)
# processed_cols['inserted_cols_array']['data'] 将是:
# np.array([[  1, -99,   2],
#           [  3, -99,   4],
#           [  5, -99,   6]])

# 3. 实例化 Insert, 沿轴 0 (行) 在索引 [0, 2] 处插入值 -77
inserter_rows = Insert(indices=[0, 2], value=-77, axis=0, source='original_array', target='inserted_rows_array')
processed_rows = inserter_rows(results.copy())
# processed_rows['inserted_rows_array']['data'] 将是 (在行0插入-77，然后在新的行2插入-77):
# np.array([[-77, -77],
#           [  1,   2],
#           [-77, -77],
#           [  3,   4],
#           [  5,   6]])
# 注意: np.insert 在指定索引之前插入。多次插入会使后续索引发生偏移。

# print("插入到列 (axis=1) 后:\n", processed_cols['inserted_cols_array']['data'])
# print("插入到行 (axis=0) 后:\n", processed_rows['inserted_rows_array']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ImageResize

~~~python
class ImageResize(BaseTransform):
    def __init__(self, size: Tuple[int, int], source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `torchvision.transforms.Resize` 将信号字典中 'data' 字段的图像数据（期望格式）调整为指定的 `size`。如果输入数据是 NumPy 数组，则会先转换为 PyTorch 张量。输出会转换回 NumPy 数组。此变换适用于图像数据，对于 NumPy 数组，其形状通常为 (C, H, W) 或 (H, W, C)，或者为 PIL 图像。

**参数**

- **size** (`Tuple[int, int]`): 期望的输出尺寸，格式为 `(高度, 宽度)`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待调整大小的信号字典（包含作为图像或类图像数组的 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已调整大小的图像 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ImageResize 已导入。
# 以及 numpy as np, torch, torchvision.transforms.Resize, 和 PIL.Image。
# from tyee.dataset.transform import ImageResize
# import numpy as np
# import torch
# from torchvision.transforms import Resize
# from PIL import Image # 其中一个转换路径需要

# 1. 准备一个包含类图像数据的示例 'results' 字典
#    示例: 一个 (通道数, 高度, 宽度) 的 NumPy 数组
results_numpy = {
    'raw_image_np': {
        'data': np.random.randint(0, 256, size=(3, 100, 150), dtype=np.uint8), # C, H, W
        'info': 'NumPy 图像数组'
    }
}
#    示例: 一个 PIL 图像 (如果您的流水线可能产生/消费这些)
# try:
#     pil_image = Image.fromarray(np.random.randint(0, 256, size=(60, 80, 3), dtype=np.uint8)) # PIL 的 H, W, C
#     results_pil = {
#         'raw_image_pil': {
#             'data': pil_image,
#             'info': 'PIL 图像'
#         }
#     }
# except ImportError:
#     results_pil = None # PIL 可能未安装

# 2. 实例化 ImageResize 变换以调整大小为 (32, 32)
resizer = ImageResize(size=(32, 32), source='raw_image_np', target='resized_image_np')

# 3. 对 NumPy 数组数据应用变换
processed_numpy = resizer(results_numpy)
# processed_numpy['resized_image_np']['data'] 将是一个形状例如 (3, 32, 32) 的 NumPy 数组。
# 通道维度的处理取决于 torchvision.transforms.Resize 对张量的行为。
# 如果输入是 (C,H,W) 张量, 输出是 (C, new_H, new_W) 张量, 然后是 (C, new_H, new_W) NumPy 数组。

# print("调整大小后的 NumPy 图像数据形状:", processed_numpy['resized_image_np']['data'].shape)

# if results_pil:
#     resizer_pil = ImageResize(size=(40, 50), source='raw_image_pil', target='resized_image_pil')
#     processed_pil = resizer_pil(results_pil)
#     # processed_pil['resized_image_pil']['data'] 将是一个形状例如 (40, 50, 3) 或 (3, 40, 50) 的 NumPy 数组
#     print("调整大小后的 PIL 图像数据形状:", processed_pil['resized_image_pil']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## Scale

~~~python
class Scale(BaseTransform):
    def __init__(self, scale_factor: float = 1.0, source: Optional[str] = None, target: Optional[str] = None):
~~~

通过将信号字典中的 'data' 字段与指定的 `scale_factor` 相乘，对其进行数值缩放。

**参数**

- **scale_factor** (`float`): 用于缩放信号数据的因子。默认为 `1.0`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待缩放的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已缩放的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Scale 已导入, numpy as np.
# from tyee.dataset.transform import Scale
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'original_signal': {
        'data': np.array([1.0, 2.5, -3.0, 4.2]),
        'channels': ['CH_A'],
        'freq': 100.0
    }
}

# 2. 实例化 Scale 变换，将数据乘以 2.0
scaler = Scale(scale_factor=2.0, source='original_signal', target='scaled_signal')

# 3. 应用变换
processed_results = scaler(results)

# 4. 'processed_results' 现在将包含 'scaled_signal'。
#    'scaled_signal' 中的 'data' 将是 [2.0, 5.0, -6.0, 8.4]。
# processed_results['scaled_signal'] 的示例内容:
# {
#   'data': np.array([2.0, 5.0, -6.0, 8.4]),
#   'channels': ['CH_A'],
#   'freq': 100.0
# }
# print(processed_results['scaled_signal'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Offset

~~~python
class Offset(BaseTransform):
    def __init__(self, offset: float | int = 0.0, source: Optional[str] = None, target: Optional[str] = None):
~~~

通过将指定的 `offset` 值添加到信号字典中的 'data' 字段，对其进行数值偏移。

**参数**

- **offset** (`float | int`): 要添加到信号数据的值。默认为 `0.0`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待偏移的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已偏移的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Offset 已导入, numpy as np.
# from tyee.dataset.transform import Offset
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'signal_to_offset': {
        'data': np.array([10, 20, 30, 40]),
        'channels': ['SensorX'],
        'freq': 50.0
    }
}

# 2. 实例化 Offset 变换，将数据加上 5
offset_adder = Offset(offset=5.0, source='signal_to_offset', target='offset_signal')

# 3. 应用变换
processed_results = offset_adder(results)

# 4. 'processed_results' 现在将包含 'offset_signal'。
#    'offset_signal' 中的 'data' 将是 [15, 25, 35, 45]。
# processed_results['offset_signal'] 的示例内容:
# {
#   'data': np.array([15, 25, 35, 45]),
#   'channels': ['SensorX'],
#   'freq': 50.0
# }
# print(processed_results['offset_signal'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Round

~~~python
class Round(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.round` 将信号字典中 'data' 字段的数值四舍五入到最接近的整数。

**参数**

- **source** (`Optional[str]`): 输入字典中的键，该键包含待四舍五入的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含已四舍五入的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Round 已导入, numpy as np.
# from tyee.dataset.transform import Round
# import numpy as np

# 1. 准备一个包含浮点数据的示例 'results' 字典
results = {
    'float_signal': {
        'data': np.array([1.2, 2.7, 3.5, 4.9, -0.3]),
        'channels': ['Values'],
    }
}

# 2. 实例化 Round 变换
rounder = Round(source='float_signal', target='rounded_signal')

# 3. 应用变换
processed_results = rounder(results)

# 4. 'processed_results' 现在将包含 'rounded_signal'。
#    'rounded_signal' 中的 'data' 将是 [1., 3., 4., 5., -0.]。
# processed_results['rounded_signal'] 的示例内容:
# {
#   'data': np.array([1., 3., 4., 5., -0.]),
#   'channels': ['Values']
# }
# print(processed_results['rounded_signal'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Log

~~~python
class Log(BaseTransform):
    def __init__(self, epsilon:float=1e-10, source: Optional[str] = None, target: Optional[str] = None):
~~~

使用 `numpy.log` 对信号字典中的 'data' 字段应用自然对数变换。在取对数之前，会向数据中添加一个 epsilon 值，以避免 `log(0)` 的问题。

**参数**

- **epsilon** (`float`): 在应用对数之前添加到数据中的一个小常量，以防止 `log(0)` 错误。默认为 `1e-10`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待变换的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（包含对数变换后的 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Log 已导入, numpy as np.
# from tyee.dataset.transform import Log
# import numpy as np

# 1. 准备一个包含正数数据的示例 'results' 字典
results = {
    'positive_signal': {
        'data': np.array([1, 10, 100, 0.1, 0.00000000001]), # 包含一个非常小的正数
        'channels': ['Intensity'],
    }
}

# 2. 实例化 Log 变换
log_transformer = Log(epsilon=1e-10, source='positive_signal', target='log_signal')

# 3. 应用变换
processed_results = log_transformer(results)

# 4. 'processed_results' 现在将包含 'log_signal'。
#    'log_signal' 中的 'data' 将是原始数据（加 epsilon）的自然对数。
#    例如, np.log(1 + 1e-10) 大约是 0。
#    np.log(10 + 1e-10) 大约是 2.302585。
#    np.log(0.00000000001 + 1e-10) 大约是 np.log(1.1e-10) 大约 -22.92
# print(processed_results['log_signal'])
~~~

[`返回顶部`](#tyeedatasettransform)

## Select

~~~python
class Select(BaseTransform):
    def __init__(self, key: Union[str, List[str]], source: Optional[str] = None, target: Optional[str] = None):
~~~

根据指定的单个键或键列表，从输入字典（通常是一个信号字典）中选择一部分条目。`BaseTransform` 中的 `source` 和 `target` 参数不直接被此变换的核心逻辑使用，因为它操作的是传递给其 `transform` 方法的字典。当在 `BaseTransform` 流水线中使用时，`source` 将决定较大结果集中的哪个字典被传递给 `transform`，而 `target` 则决定新的、筛选后的字典放置的位置。

**参数**

- **key** (`Union[str, List[str]]`): 要在输出字典中保留的单个键（字符串）或键列表（字符串列表）。
- **source** (`Optional[str]`): 如果适用，在较大结果结构中标识输入字典的键。选择逻辑本身不直接使用，但 `BaseTransform` 流水线会使用。默认为 `None`。
- **target** (`Optional[str]`): 如果适用，在较大结果结构中用于存储新字典（包含选定键）的键。选择逻辑本身不直接使用，但 `BaseTransform` 流水线会使用。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 Select 已导入。
# from tyee.dataset.transform import Select

# 1. 准备一个示例 'results' 字典 (模拟一个信号字典)
signal_dict = {
    'data': [1, 2, 3, 4, 5],
    'freq': 100.0,
    'channels': ['CH1', 'CH2'],
    'status': 'processed',
    'subject_id': 'S001'
}

# 在 BaseTransform 流水线中可能如何使用的示例：
results_pipeline_context = {
    'raw_signal_info': signal_dict
}

# 2. 实例化 Select 以仅保留 'data' 和 'freq'
selector_data_freq = Select(key=['data', 'freq'], source='raw_signal_info', target='selected_info')

# 3. 应用变换 (模拟 BaseTransform.__call__ 将如何使用它)
#    在实际的流水线中, 您会调用 selector_data_freq(results_pipeline_context)
#    这里，为清楚说明 Select 的作用，我们模拟对其 transform 方法的直接调用：
#    selected_dictionary = selector_data_freq.transform(results_pipeline_context['raw_signal_info'])
#    对于直接调用：
selected_dictionary = selector_data_freq.transform(signal_dict)

# 4. 'selected_dictionary' 将仅包含 'data' 和 'freq' 键。
# selected_dictionary 的预期内容:
# {
#   'data': [1, 2, 3, 4, 5],
#   'freq': 100.0
# }
# print(selected_dictionary)

# 选择单个键的示例
selector_subject = Select(key='subject_id')
selected_subject_dict = selector_subject.transform(signal_dict)
# selected_subject_dict 的预期内容:
# {
#   'subject_id': 'S001'
# }
# print(selected_subject_dict)
~~~

[`返回顶部`](#tyeedatasettransform)

## SlideWindow

~~~python
class SlideWindow(BaseTransform):
    def __init__(
        self,
        window_size: int,
        stride: int,
        axis: int = -1,
        source: Optional[str] = None,
        target: Optional[str] = None,
        keep_tail: bool = False
    ):
~~~

沿信号字典中 'data' 的指定轴计算滑动窗口的开始和结束索引。此变换本身不修改 'data'，而是在 `result['info']['windows']` 中添加一个窗口索引列表，并将使用的 `axis` 存储在 `result['axis']` 中。这些输出通常由后续的变换（如 `WindowExtract`）使用。

**参数**

- **window_size** (`int`): 每个滑动窗口的长度。
- **stride** (`int`): 连续窗口之间的步长或重叠。
- **axis** (`int`): 沿其滑动窗口的轴（通常是时间轴）。默认为 `-1`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含待分窗的信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，信号字典（现在包含分窗信息）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。
- **keep_tail** (`bool`): 如果为 `True`，并且最后一个窗口由于给定的步长而不能完全容纳，则会包含一个以数据最末端为结束、长度为 `window_size` 的最终窗口。这可能与前一个窗口的重叠超过 `stride` 指定的量。默认为 `False`。

**使用样例**

~~~python
# 假设 BaseTransform 和 SlideWindow 已导入, numpy as np.
# from tyee.dataset.transform import SlideWindow
# import numpy as np

# 1. 准备一个示例 'results' 字典
results = {
    'continuous_signal': {
        'data': np.arange(10).reshape(1, 10), # 1 通道, 10 时间点
        'channels': ['CH1'],
        'freq': 100.0
    }
}

# 2. 实例化 SlideWindow 以创建大小为5，步长为2的窗口
#    沿最后一个轴（时间）。
window_definer = SlideWindow(
    window_size=5,
    stride=2,
    axis=-1,
    keep_tail=False,
    source='continuous_signal',
    target='windowed_signal_info'
)

# 3. 应用变换
processed_results = window_definer(results)

# 4. 'processed_results' 现在将包含 'windowed_signal_info'。
#    - 'windowed_signal_info['data']' 仍然是原始数据。
#    - 'windowed_signal_info['info']['windows']' 将包含窗口索引。
#    - 'windowed_signal_info['axis']' 将是 -1。
#
# 对于 window_size=5, stride=2, length=10, keep_tail=False 的预期 'info']['windows']:
# [{'start': 0, 'end': 5}, {'start': 2, 'end': 7}, {'start': 4, 'end': 9}]
# (从6开始的窗口是[6,7,8,9,10]，但数据长度是10，所以end是11，如果不是keep_tail则越界。
#  如果长度是11，start=6, end=11 将是有效的。
#  长度为10时，最后一个完整窗口是 start=4, end=9。
#  如果 keep_tail=True, 且 (10-5)%2 != 0 (即 5%2=1, 为真), 它会添加:
#  {'start': 10-5=5, 'end': 10} -> [{'start': 0, 'end': 5}, {'start': 2, 'end': 7}, {'start': 4, 'end': 9}, {'start': 5, 'end': 10}]

# print(processed_results['windowed_signal_info']['info']['windows'])
# print(processed_results['windowed_signal_info']['axis'])
~~~
[`返回顶部`](#tyeedatasettransform)

## WindowExtract

~~~python
class WindowExtract(BaseTransform):
    def __init__(self, source: Optional[str] = None, target: Optional[str] = None):
~~~

根据信号字典中 `result['info']['windows']` 提供的开始/结束索引（通常由 `SlideWindow` 生成），从 'data' 数组中提取数据片段（窗口）。提取的窗口随后会沿新的第一个维度堆叠起来，生成的输出 'data' 数组形状类似于 (num_windows, channels, window_size)，具体取决于原始数据维度和分窗轴。

**参数**

- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典。此字典应包含 'data' 和 `info['windows']`（来自 `SlideWindow`）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，新的信号字典（其中 'data' 现在是堆叠的窗口）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, SlideWindow, WindowExtract 已导入, numpy as np.
# from tyee.dataset.transform import SlideWindow, WindowExtract
# import numpy as np

# 1. 准备初始数据并使用 SlideWindow 定义窗口
initial_results = {
    'continuous_signal': {
        'data': np.arange(20).reshape(2, 10), # 2 通道, 每个通道10个时间点
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}
window_definer = SlideWindow(
    window_size=4,
    stride=2,
    axis=-1, # 沿时间轴分窗
    source='continuous_signal',
    target='signal_with_windows' # SlideWindow 会修改此条目
)
results_with_window_info = window_definer(initial_results)
# results_with_window_info['signal_with_windows']['info']['windows'] 将是:
# [{'start': 0, 'end': 4}, {'start': 2, 'end': 6}, {'start': 4, 'end': 8}, {'start': 6, 'end': 10}]

# 2. 实例化 WindowExtract
#    'source' 应指向被 SlideWindow 修改的键。
window_extractor = WindowExtract(source='signal_with_windows', target='extracted_windows_data')

# 3. 应用 WindowExtract 变换
processed_results = window_extractor(results_with_window_info)

# 4. 'processed_results' 现在将包含 'extracted_windows_data'。
#    'extracted_windows_data' 中的 'data' 将是窗口的堆叠数组。
#    CH1 的原始数据: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#    CH1 的窗口: [0,1,2,3], [2,3,4,5], [4,5,6,7], [6,7,8,9]
#    CH2 的原始数据: [10,11,12,13,14,15,16,17,18,19]
#    CH2 的窗口: [10,11,12,13], [12,13,14,15], [14,15,16,17], [16,17,18,19]
#    堆叠后的 'data' 形状: (num_windows, num_channels, window_size) -> (4, 2, 4)
#
# processed_results['extracted_windows_data']['data'] 的示例内容:
# 第一个窗口 (所有通道): [[0,1,2,3], [10,11,12,13]]
# 第二个窗口 (所有通道): [[2,3,4,5], [12,13,14,15]]
# ...等等
# print(processed_results['extracted_windows_data']['data'])
# print(processed_results['extracted_windows_data']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## CWTSpectrum

~~~python
class CWTSpectrum(BaseTransform):
    def __init__(self, freqs, output_type='power', n_jobs=1, verbose=0, source=None, target=None):
~~~

使用Morlet小波通过连续小波变换（CWT）计算输入信号 'data' 的时频表示，利用 `mne.time_frequency.tfr_array_morlet`。输入数据应为2D（通道数, 时间点数）。

**参数**

- **freqs** (`array-like of float`): CWT感兴趣的频率。
- **output_type** (`str`): 要计算的输出类型。可以是 'power', 'phase', 'avg_power_itc', 'itc', 或 'complex'。默认为 `'power'`。
- **n_jobs** (`int`): 并行运行的作业数。默认为 `1`。
- **verbose** (`int | bool | str | None`): 控制详细程度。有关详细信息，请参见 `mne.time_frequency.tfr_array_morlet`。默认为 `0`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典（包含 'data' 和 'freq'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（CWT频谱作为 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 CWTSpectrum 已导入, numpy as np, mne.
# from tyee.dataset.transform import CWTSpectrum
# import numpy as np
# import mne # 用于 mne.time_frequency.tfr_array_morlet

# 1. 准备一个示例 'results' 字典
results = {
    'eeg_signal': {
        'data': np.random.randn(3, 1000), # 3 通道, 1000 时间点
        'freq': 250.0, # 采样频率
        'channels': ['CH1', 'CH2', 'CH3']
    }
}

# 2. 定义CWT感兴趣的频率
cwt_frequencies = np.arange(5, 30, 2) # 频率从 5 Hz 到 28 Hz, 步长 2 Hz

# 3. 实例化 CWTSpectrum 变换
cwt_transformer = CWTSpectrum(
    freqs=cwt_frequencies,
    output_type='power',
    source='eeg_signal',
    target='cwt_spectrum_output'
)

# 4. 应用变换
processed_results = cwt_transformer(results)

# 5. 'processed_results' 现在将包含 'cwt_spectrum_output'。
#    'cwt_spectrum_output' 中的 'data' 将是CWT功率谱，
#    形状为 (通道数, CWT频率数, 时间点数)。
#    例如, (3, len(cwt_frequencies), 1000)
# print(processed_results['cwt_spectrum_output']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## DWTSpectrum

~~~python
class DWTSpectrum(BaseTransform):
    def __init__(self, wavelet='db4', level=4, source=None, target=None):
~~~

使用 `pywt.wavedec` 计算输入信号 'data' 每个通道的离散小波变换（DWT）系数。每个通道不同分解级别的系数将被连接起来。

**参数**

- **wavelet** (`str`): 要使用的小波名称（例如，'db4', 'haar', 'sym5'）。默认为 `'db4'`。
- **level** (`int`): 分解级别。默认为 `4`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（DWT系数作为 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 DWTSpectrum 已导入, numpy as np, pywt.
# from tyee.dataset.transform import DWTSpectrum
# import numpy as np
# import pywt # 用于 pywt.wavedec

# 1. 准备一个示例 'results' 字典
results = {
    'time_series': {
        'data': np.random.randn(2, 512), # 2 通道, 512 时间点
        'freq': 128.0,
        'channels': ['SensorA', 'SensorB']
    }
}

# 2. 实例化 DWTSpectrum 变换
dwt_transformer = DWTSpectrum(
    wavelet='db4',
    level=3,
    source='time_series',
    target='dwt_coeffs_output'
)

# 3. 应用变换
processed_results = dwt_transformer(results)

# 4. 'processed_results' 现在将包含 'dwt_coeffs_output'。
#    'dwt_coeffs_output' 中的 'data' 将是一个数组，其中每一行
#    包含对应输入通道的连接后的DWT系数。
#    连接系数的长度取决于原始信号长度、小波和分解级别。
# print(processed_results['dwt_coeffs_output']['data'].shape)
~~~

[`返回顶部`](#tyeedatasettransform)

## FFTSpectrum

~~~python
class FFTSpectrum(BaseTransform):
    def __init__(
        self,
        resolution: Optional[int] = None,
        min_hz: Optional[float] = None,
        max_hz: Optional[float] = None,
        axis: int = 0, # 注意: FFT 通常应用于时间轴，通常是最后一个轴。默认为0可能不寻常，取决于数据的布局。
        sample_rate_key: str = 'freq',
        source: Optional[str] = None,
        target: Optional[str] = None
    ):
~~~

使用 `scipy.fft.rfft` 计算信号字典中 'data' 的快速傅里叶变换（FFT）幅度谱。它支持沿变换 `axis` 将信号填充或截断到指定的 `resolution`，并将结果频谱滤波到由 `min_hz` 和 `max_hz` 定义的频率范围。输出 'data' 将是幅度谱，并会添加一个包含相应频率的 'freqs' 键。

**参数**

- **resolution** (`Optional[int]`): FFT计算时沿 `axis` 的信号期望长度。如果信号较短，则进行零填充；如果较长，则进行截断。如果为 `None`，则使用原始长度。默认为 `None`。
- **min_hz** (`Optional[float]`): 输出频谱中包含的最小频率。如果为 `None`，则不应用下限。默认为 `None`。
- **max_hz** (`Optional[float]`): 输出频谱中包含的最大频率。如果为 `None`，则不应用上限。默认为 `None`。
- **axis** (`int`): 输入 'data' 中计算FFT所沿的轴（通常是时间轴）。默认为 `0`。*用户应确保这与其数据的（通道, 时间）或（时间, 通道）约定相符。*
- **sample_rate_key** (`str`): 输入信号字典中包含采样率值的键。默认为 `'freq'`。
- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 FFTSpectrum 已导入, numpy as np, scipy.fft.
# from tyee.dataset.transform import FFTSpectrum
# import numpy as np
# from scipy.fft import rfft, rfftfreq # 类功能所需

# 1. 准备示例数据
sampling_freq = 200.0
time_points = 400 # 2 秒数据
data_array = np.random.randn(3, time_points) # 3 通道, 400 时间点
results = {
    'eeg_data': {
        'data': data_array,
        'freq': sampling_freq, # sample_rate_key 的正确键名
        'channels': ['C3', 'C4', 'Cz']
    }
}

# 2. 实例化 FFTSpectrum 以获取 1 Hz 到 30 Hz 的频谱，FFT分辨率为 512 点
#    假设时间是最后一个轴，因此对于 (通道, 时间) 格式，axis 应为 -1 或 1
fft_transformer = FFTSpectrum(
    resolution=512,
    min_hz=1.0,
    max_hz=30.0,
    axis=-1, # 沿最后一个轴（时间）应用FFT
    sample_rate_key='freq',
    source='eeg_data',
    target='fft_output'
)

# 3. 应用变换
processed_results = fft_transformer(results)

# 4. 'processed_results' 将包含 'fft_output'。
#    'fft_output['data']' 将是指定频带的幅度谱。
#    'fft_output['freqs']' 将是相应的频率点。
# print("FFT 频谱数据形状:", processed_results['fft_output']['data'].shape)
# print("频率点:", processed_results['fft_output']['freqs'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ToImage

~~~python
class ToImage(BaseTransform):
    def __init__(self, length: int, width: int, resize_length_factor: float, native_resnet_size: int,
                 cmap: str = 'viridis', source: str = None, target: str = None):
~~~

将输入的信号数据（通常为2D，例如 通道 x 时间）转换为图像表示。该过程包括对比度归一化、应用颜色映射、重塑、使用 `torchvision.transforms.Resize` 调整大小、插值后再进行一次对比度归一化，最后进行ImageNet风格的归一化。输出为 `np.float32` 类型的NumPy数组。

**参数**

- **length** (`int`): 调整大小后图像的目标长度（此值会乘以 `resize_length_factor`）。
- **width** (`int`): 输入数据在初始重塑时（每个通道，在应用颜色映射之前）的宽度。这通常对应于每个通道段的时间点数或特征数。
- **resize_length_factor** (`float`): `length` 乘以该因子以确定最终调整后的图像长度。
- **native_resnet_size** (`int`): 调整大小后图像另一维度的目标尺寸（通常对应于ResNet的输入维度）。
- **cmap** (`str`): 要应用于数据的Matplotlib颜色映射的名称。默认为 `'viridis'`。
- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典（包含 'data' 和 'channels'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（图像作为 'data'）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 ToImage 已导入。
# 以及 numpy as np, torch, matplotlib as mpl, torchvision.transforms。
# from tyee.dataset.transform import ToImage
# import numpy as np
# import torch
# import matplotlib as mpl
# from torchvision import transforms

# 1. 准备示例数据 (例如, 2 通道, 20 时间点)
results = {
    'eeg_segment': {
        'data': np.random.rand(2, 20),
        'channels': ['CH1', 'CH2'],
        'freq': 100.0
    }
}

# 2. 实例化 ToImage
#    参数仅为示例；native_resnet_size 对于ResNet通常是224。
#    width 应匹配输入数据的宽度 (在此示例中为20个时间点)。
image_converter = ToImage(
    length=64,              # 图像一个维度的目标长度
    width=20,               # 应匹配输入数据的宽度 (例如时间点数)
    resize_length_factor=1.0, # 用于最终长度调整的因子
    native_resnet_size=64,  # 另一维度的目标尺寸 (例如ResNet期望的输入尺寸)
    cmap='jet',
    source='eeg_segment',
    target='image_representation'
)

# 3. 应用变换
processed_results = image_converter(results)

# 4. 'processed_results' 现在将包含 'image_representation'。
#    'data' 将是一个表示图像的 NumPy float32 数组。
#    形状将是 (3, length * resize_length_factor, native_resnet_size)，例如 (3, 64, 64)
# print(processed_results['image_representation']['data'].shape)
# print(processed_results['image_representation']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToTensor

~~~python
class ToTensor(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

使用 `torch.from_numpy()` 将信号字典中的 'data' 字段（期望为NumPy数组）转换为PyTorch张量。结果张量的数据类型是从NumPy数组推断出来的。

**参数**

- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典（包含 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（'data' 作为PyTorch张量）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 ToTensor 已导入, numpy as np, torch.
# from tyee.dataset.transform import ToTensor
# import numpy as np
# import torch

# 1. 准备一个包含NumPy数据的示例 'results' 字典
results = {
    'numpy_data_signal': {
        'data': np.array([[1, 2], [3, 4]], dtype=np.int32),
        'info': '一些数值数据'
    }
}

# 2. 实例化 ToTensor 变换
to_tensor_converter = ToTensor(source='numpy_data_signal', target='tensor_data_signal')

# 3. 应用变换
processed_results = to_tensor_converter(results)

# 4. 'processed_results' 现在将包含 'tensor_data_signal'。
#    'tensor_data_signal' 中的 'data' 将是一个PyTorch张量。
#    在这种情况下，dtype 将是 torch.int32。
# print(isinstance(processed_results['tensor_data_signal']['data'], torch.Tensor))
# print(processed_results['tensor_data_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToTensorFloat32

~~~python
class ToTensorFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为PyTorch张量，然后将其转换为 `float32` (torch.float) 类型。如果输入的 'data' 已经是PyTorch张量，则直接转换为 `float32`。如果是NumPy数组，则首先转换为张量。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToTensorFloat32, numpy as np, torch.
# from tyee.dataset.transform import ToTensorFloat32
# import numpy as np
# import torch

results = {
    'int_numpy_signal': {'data': np.array([1, 2, 3], dtype=np.int64)}
}
converter = ToTensorFloat32(source='int_numpy_signal', target='float32_tensor_signal')
processed = converter(results)
# processed['float32_tensor_signal']['data'] 是一个 torch.float32 张量
# print(processed['float32_tensor_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToTensorFloat16

~~~python
class ToTensorFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为PyTorch张量，然后将其转换为 `float16` (torch.half) 类型。如果输入的 'data' 已经是PyTorch张量，则直接转换为 `float16`。如果是NumPy数组，则首先转换为张量。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToTensorFloat16, numpy as np, torch.
# from tyee.dataset.transform import ToTensorFloat16
# import numpy as np
# import torch

results = {
    'float_numpy_signal': {'data': np.array([1.0, 2.0, 3.0], dtype=np.float64)}
}
converter = ToTensorFloat16(source='float_numpy_signal', target='float16_tensor_signal')
processed = converter(results)
# processed['float16_tensor_signal']['data'] 是一个 torch.float16 张量
# print(processed['float16_tensor_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToTensorInt64

~~~python
class ToTensorInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为PyTorch张量，然后将其转换为 `int64` (torch.long) 类型。如果输入的 'data' 已经是PyTorch张量，则直接转换为 `int64`。如果是NumPy数组，则首先转换为张量。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToTensorInt64, numpy as np, torch.
# from tyee.dataset.transform import ToTensorInt64
# import numpy as np
# import torch

results = {
    'float_numpy_signal': {'data': np.array([1.0, 2.7, 3.1], dtype=np.float32)}
}
converter = ToTensorInt64(source='float_numpy_signal', target='int64_tensor_signal')
processed = converter(results)
# processed['int64_tensor_signal']['data'] 是一个 torch.int64 张量 (值将被截断: [1, 2, 3])
# print(processed['int64_tensor_signal']['data'].dtype)
# print(processed['int64_tensor_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpy

~~~python
class ToNumpy(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

使用张量的 `.numpy()` 方法将信号字典中的 'data' 字段（期望为PyTorch张量）转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入字典中的键，该键包含信号字典（包含作为PyTorch张量的 'data'）。默认为 `None`。
- **target** (`Optional[str]`): 输出字典中的键，转换后的信号字典（'data' 作为NumPy数组）将存储在此处。如果为 `None`，则默认为 `source` 键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform 和 ToNumpy 已导入, numpy as np, torch.
# from tyee.dataset.transform import ToNumpy
# import numpy as np
# import torch

# 1. 准备一个包含PyTorch张量数据的示例 'results' 字典
results = {
    'tensor_data_signal': {
        'data': torch.tensor([[1, 2], [3, 4]], dtype=torch.float32),
        'info': '一些张量数据'
    }
}

# 2. 实例化 ToNumpy 变换
to_numpy_converter = ToNumpy(source='tensor_data_signal', target='numpy_data_signal')

# 3. 应用变换
processed_results = to_numpy_converter(results)

# 4. 'processed_results' 现在将包含 'numpy_data_signal'。
#    'numpy_data_signal' 中的 'data' 将是一个NumPy数组。
#    在这种情况下，dtype 将是 float32。
# print(isinstance(processed_results['numpy_data_signal']['data'], np.ndarray))
# print(processed_results['numpy_data_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpyFloat64

~~~python
class ToNumpyFloat64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为NumPy数组，然后将其转换为 `np.float64` 类型。如果输入的 'data' 是PyTorch张量，则首先转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToNumpyFloat64, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat64
# import numpy as np
# import torch

results = {
    'int_tensor_signal': {'data': torch.tensor([1, 2, 3], dtype=torch.int32)}
}
converter = ToNumpyFloat64(source='int_tensor_signal', target='float64_numpy_signal')
processed = converter(results)
# processed['float64_numpy_signal']['data'] 是一个 np.float64 数组
# print(processed['float64_numpy_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpyFloat32

~~~python
class ToNumpyFloat32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为NumPy数组，然后将其转换为 `np.float32` 类型。如果输入的 'data' 是PyTorch张量，则首先转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToNumpyFloat32, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat32
# import numpy as np
# import torch

results = {
    'int_tensor_signal': {'data': torch.tensor([1, 2, 3], dtype=torch.int64)}
}
converter = ToNumpyFloat32(source='int_tensor_signal', target='float32_numpy_signal')
processed = converter(results)
# processed['float32_numpy_signal']['data'] 是一个 np.float32 数组
# print(processed['float32_numpy_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpyFloat16

~~~python
class ToNumpyFloat16(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为NumPy数组，然后将其转换为 `np.float16` 类型。如果输入的 'data' 是PyTorch张量，则首先转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToNumpyFloat16, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyFloat16
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)}
}
converter = ToNumpyFloat16(source='float_tensor_signal', target='float16_numpy_signal')
processed = converter(results)
# processed['float16_numpy_signal']['data'] 是一个 np.float16 数组
# print(processed['float16_numpy_signal']['data'].dtype)
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpyInt64

~~~python
class ToNumpyInt64(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为NumPy数组，然后将其转换为 `np.int64` 类型。如果输入的 'data' 是PyTorch张量，则首先转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToNumpyInt64, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyInt64
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.1, 2.7, 3.9], dtype=torch.float32)}
}
converter = ToNumpyInt64(source='float_tensor_signal', target='int64_numpy_signal')
processed = converter(results)
# processed['int64_numpy_signal']['data'] 是一个 np.int64 数组 (值被截断: [1, 2, 3])
# print(processed['int64_numpy_signal']['data'].dtype)
# print(processed['int64_numpy_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

## ToNumpyInt32

~~~python
class ToNumpyInt32(BaseTransform):
    def __init__(self, source: str = None, target: str = None):
~~~

将信号字典中的 'data' 字段转换为NumPy数组，然后将其转换为 `np.int32` 类型。如果输入的 'data' 是PyTorch张量，则首先转换为NumPy数组。

**参数**

- **source** (`Optional[str]`): 输入信号字典的键。默认为 `None`。
- **target** (`Optional[str]`): 输出信号字典的键。默认为 `None`。

**使用样例**

~~~python
# 假设 BaseTransform, ToNumpyInt32, numpy as np, torch.
# from tyee.dataset.transform import ToNumpyInt32
# import numpy as np
# import torch

results = {
    'float_tensor_signal': {'data': torch.tensor([1.1, 2.7, 3.9], dtype=torch.float64)}
}
converter = ToNumpyInt32(source='float_tensor_signal', target='int32_numpy_signal')
processed = converter(results)
# processed['int32_numpy_signal']['data'] 是一个 np.int32 数组 (值被截断: [1, 2, 3])
# print(processed['int32_numpy_signal']['data'].dtype)
# print(processed['int32_numpy_signal']['data'])
~~~

[`返回顶部`](#tyeedatasettransform)

