# tyee.model

**内置模型列表**

**内置模型列表**

| 类名/函数名                                        | 功能描述                                                     |
| -------------------------------------------------- | ------------------------------------------------------------ |
| [`NeuralTransformer`](#neuraltransformer)         | (LaBraM) 一个用于从大规模EEG数据中学习通用表征的 Transformer 模型。 |
| [`EEGTransformer`](#eegtransformer)               | 一个核心 Transformer 特征提取器，用于处理分块后的多通道EEG时间序列。 |
| [`RelationAwareness`](#relationawareness)         | 通过融合EEG、空间位置和眼动数据来计算关系矩阵的模块。        |
| [`EffNet`](#effnet)                               | 受 EfficientNet 启发的、用于提取生理信号特征的灵活一维卷积网络骨干。 |
| [`BIOTEncoder`](#biotencoder)                       | 通过对各通道进行STFT变换后再送入Transformer处理来提取特征的模型。 |
| [`Conformer`](#conformer)                         | 一个用于EEG解码的混合卷积神经网络（CNN）- Transformer 模型。 |
| [`EcgResNet34`](#ecgresnet34)                     | 一个适用于单通道ECG分类的一维ResNet-34架构。                 |
| [`MLSTM_FCN`](#mlstm_fcn)                         | 结合了LSTM和全卷积网络（FCN）的双分支混合模型，用于多元时间序列分类。 |
| [`resnet18`](#resnet18)                         | 一个创建 `timm` ResNet-18 模型的工厂函数，用于二维图像或谱图分类。 |
| [`TwoStreamSalientModel`](#twostreamsalientmodel) | 使用EEG和EOG进行多模态睡眠分期的双流U-Net型架构。            |
| [`AutoEncoder1D`](#autoencoder1d)                 | 带有跳跃连接的一维卷积自编码器，用于信号重建或翻译任务。     |


## NeuralTransformer

NeuralTransformer是一个强大的 Transformer 模型，专为从大规模脑电图（EEG）数据中学习通用表征而设计，非常适用于各种脑机接口（BCI）任务。

- **Paper:**[Large Brain Model for Learning Generic Representations with Tremendous EEG Data in BCI](https://openreview.net/forum?id=QzTpTRVtrP)
- **Code Repository:**[LaBraM](https://github.com/935963004/LaBraM)

**初始化参数**

`NeuralTransformer` 主类通过以下参数进行初始化。像 `labram_base_patch200_200` 这样的辅助函数为该模型提供了预设的配置版本。

- **`EEG_size`** (`int`): 每个 EEG 样本的总时间点长度。默认为 `1600`。
- **`patch_size`** (`int`): 将 EEG 时间序列切分成的每个 patch 的大小。默认为 `200`。
- **`in_chans`** (`int`): 输入通道数。如果 `in_chans=1`，则使用 `TemporalConv` 作为嵌入层；否则，使用 `PatchEmbed`。默认为 `1`。
- **`out_chans`** (`int`): 当使用 `TemporalConv` 时，其输出通道数。默认为 `8`。
- **`num_classes`** (`int`): 最终分类头的输出类别数。默认为 `1000`。
- **`embed_dim`** (`int`): Transformer 模型的嵌入维度（隐藏层大小）。默认为 `200`。
- **`depth`** (`int`): Transformer 编码器层的数量。默认为 `12`。
- **`num_heads`** (`int`): 多头注意力机制中的头数。默认为 `10`。
- **`mlp_ratio`** (`float`): MLP 块（前馈网络）中隐藏层维度相对于 `embed_dim` 的比例。默认为 `4.0`。
- **`qkv_bias`** (`bool`): 是否在注意力的 q, k, v 线性层中使用偏置。默认为 `False`。
- **`drop_rate`** (`float`): 应用于 MLP 和投影层的 Dropout 概率。默认为 `0.0`。
- **`attn_drop_rate`** (`float`): 注意力权重图的 Dropout 概率。默认为 `0.0`。
- **`drop_path_rate`** (`float`): Stochastic Depth 的衰减率。默认为 `0.0`。
- **`use_abs_pos_emb`** (`bool`): 是否使用绝对位置嵌入。默认为 `True`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 4D 张量，其形状为 `(B, N, A, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `N`: 电极数量 (Number of electrodes)
- `A`: 每个电极信号被切分成的 patch 数量 (Number of patches)
- `T`: 每个 patch 的大小 (Patch size)

~~~python
# 实例化一个预设的基础模型
# 该模型期望输入样本长度为 800 (4个patch * 200点/patch)，有 4 个分类类别
model = labram_base_patch200_200(
    EEG_size=800,
    num_classes=4
)

# 创建一个符合模型输入格式的虚拟EEG数据张量
# 批量大小为 2, 电极数为 62, 每个电极有 4 个 patch, 每个 patch 200 个时间点
dummy_eeg = torch.randn(2, 62, 4, 200) 

# 将数据送入模型进行前向传播
output = model(dummy_eeg)

# 打印输出张量的形状
# 预期输出: torch.Size([2, 4]) (批量大小, 类别数)
print(output.shape)
~~~

[`返回顶部`](#tyeemodel)

## EEGTransformer

`EEGTransformer` 是一个为脑电图（EEG）信号设计的核心特征提取器。它采用标准的 Transformer 架构，首先将多通道的时间序列EEG数据切分成块（patches），然后通过多个 Transformer 编码器层来学习这些数据块之间在时间和空间维度上的深层依赖关系，最终输出能够代表原始EEG信号的特征向量序列。

- **Paper:**[EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals](https://proceedings.neurips.cc/paper_files/paper/2024/hash/4540d267eeec4e5dbd9dae9448f0b739-Abstract-Conference.html)
- **Code Repository:**[EEGPT](https://github.com/BINE022/EEGPT)

**初始化参数**

- **`img_size`** (`list`): 一个包含两个整数的列表 `[num_channels, sequence_length]`，代表输入 EEG 数据的形状。默认为 `[64, 2560]`。
- **`patch_size`** (`int`): 在时间维度上，每个 patch 的大小（时间点数量）。默认为 `64`。
- **`patch_stride`** (`int`, 可选): Patch 嵌入层的步幅。如果为 `None`，则步幅等于 `patch_size`。
- **`embed_dim`** (`int`): Transformer 模型的嵌入维度（隐藏层大小）。默认为 `768`。
- **`embed_num`** (`int`): 使用的总结性令牌（summary token）的数量。默认为 `1`。
- **`depth`** (`int`): Transformer 编码器层的数量。默认为 `12`。
- **`num_heads`** (`int`): 多头注意力机制中的头数。默认为 `12`。
- **`mlp_ratio`** (`float`): MLP 块中隐藏层维度相对于 `embed_dim` 的比例。默认为 `4.0`。
- **`qkv_bias`** (`bool`): 是否在注意力的 q, k, v 线性层中使用偏置。默认为 `True`。
- **`drop_rate`** (`float`): 应用于 MLP 和投影层的 Dropout 概率。默认为 `0.0`。
- **`attn_drop_rate`** (`float`): 注意力权重图的 Dropout 概率。默认为 `0.0`。
- **`drop_path_rate`** (`float`): Stochastic Depth 的衰减率。默认为 `0.0`。
- **`norm_layer`** (`nn.Module`): Transformer 中使用的归一化层。默认为 `nn.LayerNorm`。
- **`patch_module`** (`nn.Module`): 用于将输入信号转换为 patch 嵌入的模块。默认为 `PatchEmbed`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 3D 张量，其形状为 `(B, C, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 电极/通道数量 (Number of channels)
- `T`: 时间序列长度 (Sequence length)

该模型输出的是特征表示，而不是最终的分类结果。

~~~python
# 定义模型参数
in_channels = 62
sequence_length = 1024
patch_size = 64

# 计算 patch 数量
num_patches = sequence_length // patch_size

# 实例化模型
model = EEGTransformer(
    img_size=[in_channels, sequence_length],
    patch_size=patch_size,
    embed_dim=512,
    depth=6,
    num_heads=8
)

# 创建一个符合模型输入格式的虚拟EEG数据张量
# 批量大小为 2, 通道数为 62, 时间序列长度为 1024
dummy_eeg = torch.randn(2, in_channels, sequence_length)

# 将数据送入模型进行前向传播，提取特征
features = model(dummy_eeg)

# 打印输出特征张量的形状
# 预期输出: torch.Size([2, 16, 1, 512]) (批量大小, patch数量, embed_num, embed_dim)
print(features.shape)
~~~

[`返回顶部`](#tyeemodel)

## RelationAwareness

`RelationAwareness` 是一个用于计算关系感知特征的模块。它的核心功能是融合脑电图（EEG）特征、电极的空间位置信息以及眼动（Eye-tracking）数据，并通过一个自注意力机制来生成一个代表这些多模态信息之间关系的邻接矩阵或注意力图。

- **Paper:**[Graph to Grid: Learning Deep Representations for Multimodal Emotion Recognition](https://dl.acm.org/doi/abs/10.1145/3581783.3612074)
- **Code Repository:**[G2G-ResNet18](https://github.com/Jinminbox/G2G)

**初始化参数**

- **`head_num`** (`int`): 多头注意力机制中的头数。
- **`input_size`** (`int`): 每个电极输入的 EEG 特征维度。
- **`location_size`** (`int`): 每个电极的位置坐标的维度（例如，3D 坐标下为 3）。
- **`expand_size`** (`int`): 一个内部扩展维度，用于将 EEG、位置和眼动数据嵌入到统一的高维空间中。

**使用样例**

**重要提示**: 该模块需要三种类型的输入：EEG 特征、电极位置和眼动特征。它的输出是一个表示节点间关系的多头邻接矩阵。

~~~python
# --- 1. 定义模型参数 ---
head_num = 6
eeg_feature_dim = 5   # 对应 input_size
location_dim = 3      # 对应 location_size
expand_dim = 10       # 对应 expand_size
eye_feature_dim = 10  # 眼动特征维度

batch_size = 4
num_eeg_nodes = 62    # EEG 电极数量
num_eye_nodes = 6     # 眼动数据节点数量

# --- 2. 实例化模型 ---
model = RelationAwareness(
    head_num=head_num,
    input_size=eeg_feature_dim,
    location_size=location_dim,
    expand_size=expand_dim
)

# --- 3. 创建符合模型输入格式的虚拟数据 ---
# EEG 特征: (批量大小, 电极数, 特征维度)
eeg_features = torch.randn(batch_size, num_eeg_nodes, eeg_feature_dim)
# 电极位置: (电极数, 坐标维度) -> 扩展以匹配批量
electrode_locations = torch.randn(num_eeg_nodes, location_dim).expand(batch_size, -1, -1)
# 眼动特征: (批量大小, 眼动节点数, 特征维度)
eye_features = torch.randn(batch_size, num_eye_nodes, eye_feature_dim)

output_matrix = model(eeg_features, electrode_locations, eye_features)

~~~

[`返回顶部`](#tyeemodel)

## EffNet

`EffNet` 是一个为一维生理信号（如EEG、ECG）设计的深度卷积神经网络骨干（**backbone**），其架构灵感来源于 EfficientNet。它通过堆叠多个移动倒置瓶颈块（`MBConv`）来高效地提取时间序列数据中的深层特征。作为一个灵活的特征提取器，它的最终全连接层可以根据下游任务（如分类、回归或生成更高维的嵌入）进行替换。

- **Paper:**[SleepFM: Multi-modal Representation Learning for Sleep Across Brain Activity,ECG and Respiratory Signals](https://arxiv.org/abs/2405.17766)
- **Code Repository:**[SleepFM](https://github.com/rthapa84/sleepfm-codebase)

**初始化参数**

- **`in_channel`** (`int`): 输入信号的通道数（例如，EEG 的电极数量）。
- **`num_additional_features`** (`int`): (可选) 在最终全连接层之前要拼接的额外特征的数量。如果大于0，则模型的前向传播输入需要是一个元组 `(x, additional_features)`。默认为 `0`。
- **`depth`** (`list`): 一个整数列表，定义了模型中每个 `MBConv` 阶段所包含的瓶颈层（`Bottleneck`）数量。默认为 `[1,2,2,3,3,3,3]`。
- **`channels`** (`list`): 一个整数列表，定义了网络中每个阶段的输出通道数。默认为 `[32,16,24,40,80,112,192,320,1280]`。
- **`dilation`** (`int`): 第一个卷积层的扩张率（dilation）。默认为 `1`。
- **`stride`** (`int`): 第一个卷积层的步幅（stride）。默认为 `2`。
- **`expansion`** (`int`): `MBConv` 块内部的扩展因子。默认为 `6`。

**使用样例**

**注意**：以下示例展示了 `EffNet` 的直接用法。在默认情况下，它的最后一层输出一个单一值。然而，在更复杂的应用中，这个 `fc` 层通常被替换，以使 `EffNet` 作为特征提取骨干网络服务于不同的下游任务。

~~~python
import torch
from tyee.model import EffNet # 假设从模块中导入

# 定义模型参数
in_channels = 22      # 例如，22个EEG通道
sequence_length = 1000 # 1000个时间点

# 实例化模型 (使用默认的深度和通道配置)
model = EffNet(
    in_channel=in_channels
)
# 创建一个符合模型输入格式的虚拟EEG数据张量
# 批量大小为 4, 通道数为 22, 时间序列长度为 1000
dummy_eeg = torch.randn(4, in_channels, sequence_length)
# 将数据送入模型进行前向传播
output = model(dummy_eeg)
# 打印输出张量的形状
# 预期输出: torch.Size([4, 1]) (批量大小, 默认输出维度)
print(output.shape)
# --- 如何将其用作特征提取器 (示例) ---
# 替换掉最后一层
feature_dim = 512
model.fc = nn.Linear(model.fc.in_features, feature_dim)
# 再次前向传播，现在输出的是512维的特征
features = model(dummy_eeg)
# 预期输出: torch.Size([4, 512])
print(features.shape)
~~~

[`返回顶部`](#tyeemodel)

## BIOTEncoder

`BIOTEncoder` 是一个为多通道生理信号（如EEG）设计的特征提取器。它通过将各通道信号转换到频域，再利用 Transformer 结构来学习一个能够代表整个多通道输入的特征向量。

- **Paper:**[BIOT: Cross-data Biosignal Learning in the Wild](https://proceedings.neurips.cc/paper_files/paper/2023/hash/f6b30f3e2dd9cb53bbf2024402d02295-Abstract-Conference.html)
- **Code Repository:**[BIOT](https://github.com/ycq091044/BIOT)

**初始化参数**

- **`emb_size`** (`int`): 模型内部的嵌入维度。默认为 `256`。
- **`heads`** (`int`): Transformer 中的多头注意力头数。默认为 `8`。
- **`depth`** (`int`): Transformer 的层数。默认为 `4`。
- **`n_channels`** (`int`): 可学习的通道标识（channel token）的总数。应大于或等于数据中实际使用的通道数。默认为 `16`。
- **`n_fft`** (`int`): 短时傅里叶变换（STFT）的窗口大小。默认为 `200`。
- **`hop_length`** (`int`): STFT 的跳跃长度。默认为 `100`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 3D 张量，其形状为 `(B, C, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 信号通道数 (Number of channels)
- `T`: 时间序列长度 (Sequence length)

该模型输出的是一个固定维度的特征向量。

~~~python
# 定义模型参数
in_channels = 18
sequence_length = 2000
embedding_dimension = 256

# 实例化模型
model = BIOTEncoder(
    emb_size=embedding_dimension,
    heads=8,
    depth=4,
    n_channels=in_channels, # 确保 n_channels >= 实际通道数
    n_fft=200,
    hop_length=100
)

# 创建一个符合模型输入格式的虚拟EEG数据张量
# 批量大小为 4, 通道数为 18, 时间序列长度为 2000
dummy_eeg = torch.randn(4, in_channels, sequence_length)

# 将数据送入模型进行前向传播，提取特征
features = model(dummy_eeg)
~~~

[`返回顶部`](#tyeemodel)

## Conformer

`Conformer` 是一个专为脑电图（EEG）解码设计的混合模型，它结合了卷积神经网络（CNN）和 Transformer 的优点。模型首先使用一个卷积模块来提取局部的时空特征并生成 patch 嵌入，然后将这些嵌入序列送入一个 Transformer 编码器来捕捉长距离的依赖关系。

- **Paper:** [EEG Conformer: Convolutional Transformer for EEG Decoding and Visualization](https://ieeexplore.ieee.org/abstract/document/9991178/)
- **Code Repository:** [EEGConformer](https://github.com/eeyhsong/EEG-Conformer)

**初始化参数**

- **`n_outputs`** (`int`): 最终分类层的输出类别数。
- **`n_chans`** (`int`): 输入的 EEG 通道数量。
- **`n_times`** (`int`): 输入信号的时间点数量。当 `final_fc_length="auto"` 时必须提供。
- **`n_filters_time`** (`int`): 时间卷积核的数量，该值也定义了 Transformer 的嵌入维度。默认为 `40`。
- **`filter_time_length`** (`int`): 时间卷积核的长度。默认为 `25`。
- **`pool_time_length`** (`int`): 时间池化层的核长度。默认为 `75`。
- **`pool_time_stride`** (`int`): 时间池化层的步幅。默认为 `15`。
- **`drop_prob`** (`float`): 初始卷积模块中的 Dropout 概率。默认为 `0.5`。
- **`att_depth`** (`int`): Transformer 编码器的层数。默认为 `6`。
- **`att_heads`** (`int`): Transformer 中的多头注意力头数。默认为 `10`。
- **`att_drop_prob`** (`float`): Transformer 注意力层中的 Dropout 概率。默认为 `0.5`。
- **`final_fc_length`** (`int` 或 `str`): 最终全连接层输入维度的大小。可设为 `"auto"` 使其自动计算。默认为 `"auto"`。
- **`return_features`** (`bool`): 如果为 `True`，模型将返回分类层之前的特征。默认为 `False`。
- **`activation`** (`nn.Module`): 卷积模块中使用的激活函数。默认为 `nn.ELU`。
- **`activation_transfor`** (`nn.Module`): Transformer 前馈网络中使用的激活函数。默认为 `nn.GELU`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 3D 张量，其形状为 `(B, C, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 信号通道数 (Number of channels)
- `T`: 时间序列长度 (Sequence length)

~~~python
# 定义模型参数
n_outputs = 4
n_chans = 22
n_times = 1000

# 实例化模型
model = Conformer(
    n_outputs=n_outputs,
    n_chans=n_chans,
    n_times=n_times,
    att_depth=6,
    att_heads=10,
    final_fc_length='auto'
)

# 创建一个符合模型输入格式的虚拟EEG数据张量
# 批量大小为 8, 通道数为 22, 时间序列长度为 1000
dummy_eeg = torch.randn(8, n_chans, n_times)

# 将数据送入模型进行前向传播
output = model(dummy_eeg)

# 打印输出张量的形状
# 预期输出: torch.Size([8, 4]) (批量大小, 类别数)
print(output.shape)
~~~

[`返回顶部`](#tyeemodel)

## EcgResNet34

`EcgResNet34` 是一个为一维心电图（ECG）信号设计的深度残差网络。它将经典的 ResNet-34 架构改编为一维形式，通过堆叠多个残差块（`BasicBlock`）来学习时间序列信号中的特征，适用于 ECG 信号的分类任务。

- **Paper:**[Diagnosis of Diseases by ECG Using Convolutional Neural Networks](https://www.hse.ru/en/edu/vkr/368722189)
- **Code Repository:**[ECGResNet34](https://github.com/lxdv/ecg-classification)

**初始化参数**

- **`layers`** (`tuple`): 一个包含4个整数的元组，分别定义了 ResNet 四个阶段中残差块的数量。默认为 `(1, 5, 5, 5)`。
- **`num_classes`** (`int`): 最终分类层的输出类别数。默认为 `1000`。
- **`zero_init_residual`** (`bool`): 如果为 `True`，则将每个残差块中最后一个批归一化（BatchNorm）层的权重初始化为零。默认为 `False`。
- **`groups`** (`int`): 卷积层中的分组数。`BasicBlock` 只支持 `1`。默认为 `1`。
- **`width_per_group`** (`int`): 每组的宽度。`BasicBlock` 只支持 `64`。默认为 `64`。
- **`replace_stride_with_dilation`** (`list`, 可选): 一个布尔值列表，决定是否用扩张卷积替换后续阶段的步幅卷积。
- **`norm_layer`** (`nn.Module`): 模型中使用的归一化层。默认为 `nn.BatchNorm1d`。
- **`block`** (`nn.Module`):构成网络的残差块类型。默认为 `BasicBlock`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 3D 张量，其形状为 `(B, 1, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 信号通道数 (该模型固定为 **1** )
- `T`: 时间序列长度 (Sequence length)

~~~python
# 定义模型参数
num_classes = 5
sequence_length = 2048

# 实例化模型
# 使用类似 ResNet-34 的层配置
model = EcgResNet34(
    layers=(3, 4, 6, 3), # 这是 ResNet-34 的标准配置
    num_classes=num_classes
)

# 创建一个符合模型输入格式的虚拟ECG数据张量
# 批量大小为 8, 通道数为 1, 时间序列长度为 2048
dummy_ecg = torch.randn(8, 1, sequence_length)

# 将数据送入模型进行前向传播
output = model(dummy_ecg)

# 打印输出张量的形状
# 预期输出: torch.Size([8, 5]) (批量大小, 类别数)
print(output.shape)
~~~

[`返回顶部`](#tyeemodel)

## AutoEncoder1D

`AutoEncoder1D` 是一个为一维时序信号设计的卷积自编码器模型。它采用经典的编码器-解码器（Encoder-Decoder）架构，并加入了跳跃连接（skip connections）来帮助保留和重建细节信息。该模型适用于信号重建、去噪或将一种信号转换为另一种信号（信号翻译）等任务。

- **Paper:**[FingerFlex: Inferring Finger Trajectories from ECoG signals](https://arxiv.org/abs/2211.01960)
- **Code Repository:**[FingerFlex](https://github.com/Irautak/FingerFlex)

**初始化参数**

- **`n_electrodes`** (`int`): 输入信号的电极/通道数量。默认为 `30`。
- **`n_freqs`** (`int`): 每个电极/通道的特征维度（例如，经过小波或傅里叶变换后的频率波数）。默认为 `16`。
- **`n_channels_out`** (`int`): 解码器最终输出的通道数。默认为 `21`。
- **`channels`** (`list`): 一个整数列表，定义了编码器每个阶段的输出通道数。默认为 `[8, 16, 32, 32]`。
- **`kernel_sizes`** (`list`): 一个整数列表，定义了编码器中每个卷积块的卷积核大小。默认为 `[3, 3, 3]`。
- **`strides`** (`list`): 一个整数列表，定义了编码器中每个卷积块的下采样步幅。默认为 `[4, 4, 4]`。
- **`dilation`** (`list`): 一个整数列表，定义了编码器中每个卷积块的扩张率。默认为 `[1, 1, 1]`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 4D 张量，其形状为 `(B, E, F, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `E`: 电极/通道数量 (Number of electrodes/channels)
- `F`: 每个通道的特征/频率数 (Number of features/frequencies per channel)
- `T`: 时间序列长度 (Sequence length)

~~~python
# 定义模型参数
n_electrodes = 62
n_freqs = 16
n_channels_out = 5  # 例如，解码为5个手指的运动轨迹
sequence_length = 1024

# 实例化模型
model = AutoEncoder1D(
    n_electrodes=n_electrodes,
    n_freqs=n_freqs,
    n_channels_out=n_channels_out,
    channels=[16, 32, 64],
    strides=[4, 4, 2]
)

# 创建一个符合模型输入格式的虚拟数据张量
# 批量大小为 4, 电极数为 62, 频率数为 16, 时间序列长度为 1024
dummy_input = torch.randn(4, n_electrodes, n_freqs, sequence_length)

# 将数据送入模型进行前向传播
output_signal = model(dummy_input)

# 打印输出张量的形状
# 预期输出: torch.Size([4, 5, 1024]) (批量大小, 输出通道数, 原始时间序列长度)
print(output_signal.shape)
~~~

[`返回顶部`](#tyeemodel)

## MLSTM_FCN

`MLSTM_FCN` 是一个用于多元时间序列分类的深度学习模型。它采用了一个双分支的混合架构，结合了长短期记忆网络（LSTM）和全卷积网络（FCN）的优点，以同时捕捉时间序列的长期依赖性和局部特征。

- **Paper:**[Multivariate LSTM-FCNs for time series classification](https://www.sciencedirect.com/science/article/abs/pii/S0893608019301200)
- **Code Repository:**[MLSTM-FCN](https://github.com/titu1994/MLSTM-FCN)

**初始化参数**

- **`max_nb_variables`** (`int`): 输入多元时间序列的变量数量（即通道数）。
- **`max_timesteps`** (`int`): 输入时间序列的长度（时间点数量）。
- **`nb_class`** (`int`): 最终分类层的输出类别数。
- **`lstm_units`** (`int`): LSTM 层中的隐藏单元数量。默认为 `8`。
- **`dropout_rate`** (`float`): 应用于 LSTM 路径输出的 Dropout 概率。默认为 `0.8`。

**使用样例**

**重要提示**: 该模型期望的输入张量是一个 3D 张量，其形状为 `(B, C, T)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 变量/通道数量 (Number of variables/channels)
- `T`: 时间序列长度 (Sequence length)

~~~python
# 定义模型参数
num_variables = 5  # 例如，5个变量
num_timesteps = 640 # 640个时间点
num_classes = 9

# 实例化模型
model = MLSTM_FCN(
    max_nb_variables=num_variables,
    max_timesteps=num_timesteps,
    nb_class=num_classes
)

# 创建一个符合模型输入格式的虚拟数据张量
# 批量大小为 16, 变量数为 5, 时间序列长度为 640
dummy_input = torch.randn(16, num_variables, num_timesteps)

# 将数据送入模型进行前向传播
output = model(dummy_input)

# 打印输出张量的形状
# 预期输出: torch.Size([16, 9]) (批量大小, 类别数)
print(output.shape)
~~~

[`返回顶部`](#tyeemodel)

## TwoStreamSalientModel

`TwoStreamSalientModel` 是一个为多模态生理信号（特别是EEG和EOG）设计的双流深度学习模型。该模型的核心是一个复杂的U-Net型架构，每个输入流（EEG和EOG）都由一个独立的、包含编码器-解码器和多尺度特征提取（MSE）模块的分支（`Branch`）进行处理。两个分支的特征在后期进行融合，并通过一个注意力机制来生成最终的序列分类结果，非常适用于睡眠分期等任务。

- **Paper:**[SalientSleepNet: Multimodal Salient Wave Detection Network for Sleep Staging](https://arxiv.org/abs/2105.13864)
- **Code Repository:**[SalientSleepNet](https://github.com/ziyujia/SalientSleepNet)

**初始化参数**

该模型通过一个单独的 `config` 字典进行初始化，该字典包含了定义整个网络架构的所有超参数。

- `config`

   (

  ```
  dict
  ```

  ): 包含模型配置的字典。其主要键值包括：

  - **`sleep_epoch_len`**: 单个睡眠周期的长度（以采样点计）。

  - **`preprocess`**: 预处理相关的配置，如 `sequence_epochs` (输入序列包含的周期数)。

  - `train`

    : 训练和网络结构相关的参数，例如：

    - `filters`: 定义U-Net各阶段通道数的列表。
    - `kernel_size`: 卷积核大小。
    - `pooling_sizes`: 编码器中各阶段池化层的大小。
    - `dilation_sizes`: MSE模块中的扩张率列表。
    - `u_depths`: U-Net单元的深度。
    - `u_inner_filter`: U-Net单元内部的滤波器数量。
    - `mse_filters`: MSE模块中的滤波器数量。

**使用样例**

**重要提示**: 该模型需要两个独立的输入张量，`x_eeg` 和 `x_eog`。每个张量的形状都应为 `(B, 1, T, 1)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 输入通道数 (固定为 **1** )
- `T`: 总时间序列长度 (`sequence_epochs * sleep_epoch_len`)
- `W`: 宽度维度 (固定为 **1** )

该模型输出一个序列预测结果，每个时间步对应一个分类概率分布。

~~~python
# 1. 定义一个示例配置字典
sample_config = {
    'sleep_epoch_len': 3000,
    'preprocess': {
        'sequence_epochs': 20
    },
    'train': {
        'filters': [16, 32, 64, 128, 256],
        'kernel_size': 5,
        'pooling_sizes': [10, 8, 6, 4],
        'dilation_sizes': [1, 2, 3, 4],
        'activation': 'relu',
        'u_depths': [4, 4, 4, 4],
        'u_inner_filter': 16,
        'mse_filters': [8, 16, 32, 64, 128],
        'padding': 'same'
    }
}

# 2. 实例化模型
model = TwoStreamSalientModel(sample_config)

# 3. 创建符合模型输入格式的虚拟数据张量
batch_size = 2
total_timesteps = sample_config['preprocess']['sequence_epochs'] * sample_config['sleep_epoch_len']
dummy_eeg = torch.randn(batch_size, 1, total_timesteps, 1)
dummy_eog = torch.randn(batch_size, 1, total_timesteps, 1)

# 4. 将数据送入模型进行前向传播
predictions = model(dummy_eeg, dummy_eog)

# 5. 打印输出张量的形状
# 预期输出: torch.Size([2, 20, 5]) (批量大小, 序列长度, 类别数)
# 其中类别数在模型中硬编码为5
print(predictions.shape)
~~~

[`返回顶部`](#tyeemodel)

## resnet18

这是一个便捷函数，它使用 `timm` 库来创建一个 **ResNet-18** 模型。该函数简化了模型的实例化过程，并支持加载预训练权重以及自定义的检查点，非常适用于图像或二维表征（如谱图）的分类任务。

- **Paper:** [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- **Code Repository:** [pytorch-image-models (timm)](https://github.com/rwightman/pytorch-image-models)

**函数参数**

- **`num_classes`** (`int`): 最终分类层的输出类别数。
- **`pretrained`** (`bool`): 如果为 `True`，则加载在 ImageNet 上预训练的权重。默认为 `True`。
- **`pretrained_cfg_overlay`** (`dict`, 可选): 一个字典，用于覆盖 `timm` 模型默认的预训练配置（例如，修改输入尺寸或均值/标准差）。默认为 `None`。
- **`checkpoint_path`** (`str`, 可选): 一个本地检查点文件（`.pth`）的路径。如果提供，将从该文件加载模型权重。默认为 `None`。

**使用样例**

**重要提示**: 由 `timm` 创建的 ResNet-18 模型期望的输入张量是一个 4D 张量，其形状为 `(B, 3, H, W)`，分别代表：

- `B`: 批量大小 (Batch size)
- `C`: 输入通道数 (通常为 **3**，代表RGB图像)
- `H`: 图像高度 (Height)
- `W`: 图像宽度 (Width)

~~~python
# 定义模型参数
num_classes = 10 # 假设有10个手势类别

# 调用函数创建 ResNet-18 模型
# pretrained=True 将加载 ImageNet 预训练权重
model = resnet18(
    num_classes=num_classes,
    pretrained=True
)

# 创建一个符合模型输入格式的虚拟数据张量
# 批量大小为 4, 3个通道 (RGB), 图像尺寸为 224x224
dummy_input = torch.randn(4, 3, 224, 224)

# 将数据送入模型进行前向传播
output = model(dummy_input)

# 打印输出张量的形状
# 预期输出: torch.Size([4, 10]) (批量大小, 类别数)
print(output.shape)
~~~

[`返回顶部`](#tyeemodel)

