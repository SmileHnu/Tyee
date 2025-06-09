# tyee.dataset.split

`tyee.dataset.split`模块提供丰富的数据集划分方法，供用户选择不同的方式。

| 划分方法名称                                          | 功能作用                                                     |
| ----------------------------------------------------- | ------------------------------------------------------------ |
| [`BaseSplit`](#basesplit)                            | 所有切分策略的基类，定义了通用接口。                         |
| [`HoldOut`](#holdout)                               | 对整个数据集进行一次简单的训练集/验证集切分。                |
| [`HoldOutCross`](#holdoutcross)                     | 基于指定的列（如 'subject_id'）对整个数据集进行分组，然后按组切分训练集/验证集，确保同一组的数据不跨集。 |
| [`HoldOutGroupby`](#holdoutgroupby)                 | 层级式留出法：首先按 `base_group` (如 'subject_id') 分组，然后在每个 `base_group` 内部再按 `group_by` (如 'trial_id') 的唯一单元切分训练/验证数据，并聚合结果。 |
| [`HoldOutPerSubject`](#holdoutpersubject)           | 针对*每个被试*独立地将其数据切分为训练集和验证集。`split` 方法会遍历所有被试的切分结果。 |
| [`HoldOutPerSubjectET`](#holdoutpersubjectet)       | 针对*每个被试*独立地将其数据切分为训练集、验证集*和测试集*。`split` 方法会遍历所有被试的切分结果。 |
| [`HoldOutPerSubjectCross`](#holdoutpersubjectcross) | 针对*每个被试*，将其内部的（由`group_by`定义的）组（例如试验）切分为训练和验证两部分。`split` 方法会遍历所有被试的这种内部组切分。 |
| [`KFold`](#kfold)                                   | 对整个数据集的样本进行标准的K折交叉验证切分。                |
| [`KFoldCross`](#kfoldcross)                         | K折交叉验证，但切分的对象是按 `group_by` 列定义的组（例如被试ID），确保同一组的所有数据始终在同一折中（训练或验证）。 |
| [`KFoldGroupby`](#kfoldgroupby)                     | 层级式K折：首先按 `base_group` 分组，然后在每个 `base_group` 内部的每个 `group_by` 单元内，对其*样本*应用K折切分。最终的K个折是这些样本级切分的聚合。 |
| [`KFoldPerSubject`](#kfoldpersubject)               | 针对*每个被试*，独立地对其数据样本应用K折交叉验证。`split` 方法会遍历所有被试的所有折。 |
| [`KFoldPerSubjectCross`](#kfoldpersubjectcross)     | 针对*每个被试*，将其内部的（由`group_by`定义的）组（例如试验）进行K折交叉验证切分。`split` 方法会遍历所有被试的这种内部组的K折切分。 |
| [`LeaveOneOut`](#leaveoneout)                       | 留一组交叉验证：每次迭代留出一个由 `group_by` 定义的组作为验证集，其余组作为训练集。 |
| [`LeaveOneOutEAndT`](#leaveoneouteandt)             | 留一法变体：每次迭代指定一个组为测试组 (`test_group`)，其前一个组为验证组 (`val_group`)，其余为训练组。 |
| [`LosoRotatingCrossSplit`](#losorotatingcrosssplit) | 复杂的轮转切分：首先对组进行粗略K折切分，然后对每次粗略切分中"剩余"的组进行轮转，以依次分配给验证集和测试集，其余构成训练集，从而生成多个训练/验证/测试切分。 |
| [`NoSplit`](#nosplit)                                | 不执行任何实际的数据切分，直接将输入的训练集、验证集（可选）、测试集（可选）原样返回。适用于已预先切分的数据或在完整数据集上评估的场景。 |



## BaseSplit

~~~python
class BaseSplit:
    def __init__(
        self,
        split_path: str,
        **kwargs
    ) -> None:
~~~

这是用于定义数据集切分策略的基类。它为使用路径初始化切分机制提供了一个通用结构，并包含一个检查此路径有效性的方法。子类应实现实际的 `split` 逻辑。

**参数**

- **split_path** (`str`): 指向切分信息（例如，定义训练/验证/测试切分的CSV文件）所在或将要存储的目录的路径。
- ***\*kwargs**: 子类可能使用的其他关键字参数。

**方法**

~~~python
def check_split_path(self) -> bool:
~~~

检查 `split_path` 目录是否存在并且至少包含一个CSV文件。它会记录有关其发现的信息。

*返回:*

- `bool`: 如果路径存在且至少包含一个CSV文件，则为 `True`，否则为 `False`。

~~~python
def split(
    self, 
    dataset: BaseDataset,
    val_dataset: BaseDataset = None,
    test_dataset: BaseDataset = None,
) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
~~~

此方法旨在由子类覆盖以实现特定的数据集切分逻辑。它应定义如何将给定的 `dataset`（以及可选的 `val_dataset` 和 `test_dataset`）划分为训练集、验证集和测试集。

*参数:*

- **dataset** (`BaseDataset`): 要切分的主要数据集。
- **val_dataset** (`BaseDataset`, optional): 可选的、独立的验证数据集。
- **test_dataset** (`BaseDataset`, optional): 可选的、独立的测试数据集。

*产出 (Yields):*

- `Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]`: 一个生成器，通常应产出元组，代表 `(训练数据集, 验证数据集, 测试数据集)`。

*引发:*

- `NotImplementedError`: 此基类方法必须由子类覆盖。

`BaseSplit` 是一个抽象基类，通常不直接实例化，而是用于被继承.

[`返回顶部`](#tyeedatasetsplit)
## Hold Out

### HoldOut

~~~python
class HoldOut(BaseSplit):
    def __init__(
        self,
        val_size: float = 0.2,
        shuffle: bool = False,
        stratify: str = None,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现留出法验证策略，将数据集切分为单个训练集和验证集。如果 `split_path` 已提供并且包含现有的 `train.csv` 和 `val.csv` 文件，则使用这些切分；否则，将根据数据集的 `info` DataFrame 生成新的切分，并保存到 `split_path`。可选的测试集可以被传递，但此方法不会主动对其进行切分。

**参数**

- **val_size** (`float`): 数据集中包含在验证集中的比例。应介于0.0和1.0之间。默认为 `0.2`。
- **shuffle** (`bool`): 切分前是否打乱数据。默认为 `False`。
- **stratify** (`str`, optional): 如果不是 `None`，则以分层方式切分数据，使用此字符串作为数据集中 `info` DataFrame 中用于分层的列名。默认为 `None`。
- **random_state** (`Union[int, None]`): 控制在应用切分前对数据进行的打乱操作。传递一个整数以确保多次函数调用的可复现输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储 `train.csv` 和 `val.csv`（定义切分）的目录路径。如果路径不存在或不包含这些文件，则会生成新的切分并保存。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame。
# 若要进行分层抽样 (stratify='label')，'dataset.info' 应包含名为 'label' 的列。

hold_out_splitter = HoldOut(
    val_size=0.25,
    shuffle=True,
    stratify='label', # 假设 dataset.info 中存在 'label' 列
    random_state=42,
    split_path='实际存放切分文件的路径' # 替换为实际的目录路径
)

# 'split' 方法是一个生成器。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已更新以反映各自的切分。
# 如果没有向 .split() 方法传递 test_dataset，则 'test_set_passthrough' 将为 None。
for train_set, val_set, test_set_passthrough in hold_out_splitter.split(dataset):
    # 使用 train_set 和 val_set 进行训练和验证。
    # 例如: print(f"训练集大小: {len(train_set.info)}, 验证集大小: {len(val_set.info)}")
    pass
~~~

[`返回顶部`](#tyeedatasetsplit)

### HoldOutCross

~~~python
class HoldOutCross(BaseSplit):
    def __init__(
        self,
        group_by: str,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种基于特定分组列的留出交叉验证策略（例如，按 'subject_id', 'session_id' 分组）。此方法将数据集划分为训练集和验证集，确保属于同一组的所有数据点（例如，来自同一被试的所有试验）完全位于训练集或验证集中，从而防止同一组数据在切分集之间发生泄漏。如果 `split_path` 已提供并且包含现有的 `train.csv` 和 `val.csv` 文件，则使用这些切分；否则，将根据数据集中 `info` DataFrame 里的唯一组ID生成新的切分，并保存到 `split_path`。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于在切分前对数据进行分组的列名（例如 'trial_id', 'session_id', 'subject_id'）。
- **val_size** (`float`): 要包含在验证集中的唯一组的比例。此值应介于0.0和1.0之间。默认为 `0.2`。
- **shuffle** (`bool`): 在将唯一组ID切分为训练组和验证组之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[None, int]`): 控制组ID的打乱。传递一个整数以确保多次函数调用之间的切分可复现。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储 `train.csv` 和 `val.csv`（基于原始数据集 `info` 定义切分）的目录路径。如果路径不存在或不包含这些文件，则会生成新的切分并保存。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'group_by' 指定的列 (例如, 'subject_id')。

# 示例：按 'subject_id' 切分，将20%的被试分入验证集。
hold_out_cross_splitter = HoldOutCross(
    group_by='subject_id', # 按被试ID分组数据
    val_size=0.2,
    shuffle=True,
    random_state=42,
    split_path='存放分组切分文件的路径' # 替换为实际的目录路径
)

# 'split' 方法是一个生成器。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已更新以反映按组的切分。
# 如果没有向 .split() 方法传递 test_dataset，则 'test_set_passthrough' 将为 None。
for train_set, val_set, test_set_passthrough in hold_out_cross_splitter.split(dataset):
    # 使用 train_set 和 val_set 进行训练和验证。
    # 来自任何给定 subject_id 的所有数据将完全位于 train_set 或 val_set 中。
    # 例如: print(f"训练集 .info 长度: {len(train_set.info)}, 验证集 .info 长度: {len(val_set.info)}")
    pass
~~~

[`返回顶部`](#tyeedatasetsplit)

### HoldOutGroupby

~~~python
class HoldOutGroupby(BaseSplit):
    def __init__(
        self,
        group_by: str,
        base_group: str = "subject_id",
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种具有层级分组的留出验证策略。数据集首先按一个 `base_group`（例如 'subject_id'）进行一级分组，然后在每个基本组内，再按一个次级的 `group_by`（例如 'trial_id' 或 'session_id'）进行分组切分，从而划分为训练集和验证集。这确保了在同一个基本组内，属于同一个次级组的所有数据点完全位于训练集或验证集中。切分对每个基本组独立执行，然后将所有基本组产生的训练集和验证集分别聚合起来。如果 `split_path` 已提供并且包含现有的 `train.csv` 和 `val.csv` 文件，则使用这些切分；否则，将生成新的切分并保存。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于次级分组的列名（例如 'trial_id', 'session_id'）。训练/验证切分将在这个组级别上，在每个 `base_group` 内部进行。
- **base_group** (`str`): 数据集 `info` DataFrame 中用于主要、较粗粒度分组的列名（例如 'subject_id'）。切分过程将针对此基本组中的每个唯一值进行迭代。默认为 `"subject_id"`。
- **val_size** (`float`): 在每个 `base_group` 内，要包含在验证集中的次级组（由 `group_by` 定义）的比例。此值应介于0.0和1.0之间。默认为 `0.2`。
- **shuffle** (`bool`): 在切分次级组（在每个基本组内）之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[int, None]`): 控制次级组的打乱。传递一个整数以确保切分的可复现性。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储 `train.csv` 和 `val.csv`（定义聚合后的切分）的目录路径。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'base_group' (例如 'subject_id') 
# 和 'group_by' (例如 'session_id') 指定的列。

# 示例：对于每个被试 ('base_group')，切分其会话 ('group_by')，
# 使得每个被试20%的会话进入验证集。
hierarchical_hold_out_splitter = HoldOutGroupby(
    group_by='session_id',    # 在每个被试内部切分会话
    base_group='subject_id',  # 遍历所有被试
    val_size=0.20,
    shuffle=True,
    random_state=42,
    split_path='存放层级切分文件的路径' # 替换为实际的目录路径
)

# 'split' 方法是一个生成器。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已更新以反映聚合后的层级切分。
for train_set, val_set, test_set_passthrough in hierarchical_hold_out_splitter.split(dataset):
    # 使用 train_set 和 val_set 进行训练和验证。
    # 来自任何给定 session_id (在某个 subject_id 内) 的所有数据将完全位于该被试切分部分的 train_set 或 val_set 中。
    # 例如: print(f"聚合训练集大小: {len(train_set.info)}, 聚合验证集大小: {len(val_set.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### HoldOutPerSubject

~~~python
class HoldOutPerSubject(BaseSplit):
    def __init__(
        self,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种留出验证策略，其中*每个被试*的数据被独立地切分为训练集和验证集。这意味着对于数据集中的每一个被试，他们数据的一部分（`val_size`）被留出用于验证，其余部分用于训练。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_subject_X.csv`, `val_subject_X.csv`），则对相应被试使用这些切分；否则，将根据数据集中 `info` DataFrame 里每个被试的数据生成新的切分，并保存到 `split_path`。`split` 方法随后可以为所有被试或特定请求的被试产出这些训练/验证数据对。

**参数**

- **val_size** (`float`): 每个被试数据中包含在其各自验证集内的比例。应介于0.0和1.0之间。默认为 `0.2`。
- **shuffle** (`bool`): 在切分每个被试的数据之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[int, None]`): 控制应用于每个被试数据的打乱操作。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每个被试切分文件（`train_subject_X.csv`, `val_subject_X.csv`）的目录路径。如果为 `None`，行为可能取决于 `BaseSplit` 或内部处理，但通常需要此路径来进行保存/加载。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含一个 'subject_id' 列。

# 示例：对于每个被试，将其25%的数据留作验证集。
per_subject_splitter = HoldOutPerSubject(
    val_size=0.25,
    shuffle=True,
    random_state=42,
    split_path='存放每个被试切分文件的路径' # 替换为实际的目录路径
)

# 遍历所有被试的切分：
# 'train_set' 和 'val_set' 将是 BaseDataset 实例，其 '.info' 属性
# 特定于某个被试的训练/验证部分。
# 'test_set_passthrough' 通常是 None 或一个被一同传递的通用测试集。
# for train_set, val_set, test_set_passthrough in per_subject_splitter.split(dataset):
#     # 这个循环会为基于切分文件或 dataset.info 找到的每个被试运行。
#     # 例如: print(f"正在处理被试: {train_set.info['subject_id'].iloc[0]}")
#     # print(f"  训练集大小: {len(train_set.info)}, 验证集大小: {len(val_set.info)}")
#     pass

# 示例：获取特定被试的切分 (例如, 被试 'S01')
# 确保 'S01' 是数据集中或切分文件中存在的有效被试ID。
specific_subject_id = 'S01' # 替换为实际的被试ID字符串
for train_set_s01, val_set_s01, test_set_s01 in per_subject_splitter.split(dataset, subject=specific_subject_id):
    # 这个循环将仅为被试 'S01' 运行。
    # print(f"正在处理特定被试: {specific_subject_id}")
    # print(f"  训练集大小: {len(train_set_s01.info)}, 验证集大小: {len(val_set_s01.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### HoldOutPerSubjectET

~~~python
class HoldOutPerSubjectET(BaseSplit):
    def __init__(
        self,
        val_size: float = 0.2,
        test_size: float = 0.2,
        stratify: str = None,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种留出策略，其中*每个被试*的数据被独立地切分为训练集、验证集*和测试集*。`val_size` 和 `test_size` 的总和必须小于1.0。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_subject_X.csv`, `val_subject_X.csv`, `test_subject_X.csv`），则使用这些切分；否则，将为每个被试生成新的切分并保存。`split` 方法随后可以为所有被试或特定请求的被试产出这些训练/验证/测试数据三元组。如果提供了 `stratify` 列名，则可以在切分过程中应用分层抽样。

**参数**

- **val_size** (`float`): 每个被试数据中包含在其各自验证集内的比例。默认为 `0.2`。
- **test_size** (`float`): 每个被试数据中包含在其各自测试集内的比例。默认为 `0.2`。
- **stratify** (`str`, optional): 如果不是 `None`，则每个被试的数据将以分层方式切分，使用此字符串作为数据集中 `info` DataFrame 中用于分层的列名。默认为 `None`。
- **shuffle** (`bool`): 在切分每个被试的数据之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[int, None]`): 控制应用于每个被试数据的打乱操作。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每个被试切分文件（`train_subject_X.csv`, `val_subject_X.csv`, `test_subject_X.csv`）的目录路径。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含一个 'subject_id' 列，
# 以及在使用了 stratify='label' 时的 'label' 列。

# 示例：对于每个被试，创建训练/验证/测试切分 (例如, 60/20/20)。
per_subject_tvt_splitter = HoldOutPerSubjectET(
    val_size=0.2,
    test_size=0.2,
    stratify='label', # 可选: 在每个被试的数据内按 'label' 分层
    shuffle=True,
    random_state=42,
    split_path='存放每个被试TVT切分文件的路径' # 替换为实际的目录路径
)

# 遍历所有被试的切分：
# for train_s, val_s, test_s in per_subject_tvt_splitter.split(dataset):
#     # 这个循环会为每个被试运行。
#     # 'train_s', 'val_s', 'test_s' 是针对一个被试的 BaseDataset 实例。
#     # print(f"正在处理被试: {train_s.info['subject_id'].iloc[0]}")
#     # print(f"  训练集: {len(train_s.info)}, 验证集: {len(val_s.info)}, 测试集: {len(test_s.info)}")
#     pass

# 示例：获取特定被试的切分 (例如, 被试 'S02')
specific_subject_id_et = 'S02' # 替换为实际的被试ID字符串
for train_s02, val_s02, test_s02 in per_subject_tvt_splitter.split(dataset, subject=specific_subject_id_et):
    # 这个循环将仅为被试 'S02' 运行。
    # print(f"正在处理特定被试: {specific_subject_id_et}")
    # print(f"  训练集: {len(train_s02.info)}, 验证集: {len(val_s02.info)}, 测试集: {len(test_s02.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### HoldOutPerSubjectCross

~~~python
class HoldOutPerSubjectCross(BaseSplit):
    def __init__(
        self,
        group_by: str,
        val_size: float = 0.2,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种针对每个被试独立应用的留出验证策略，其中*在每个被试的数据内部*的切分是基于指定的分组列（例如 'trial_id', 'session_id'）。这确保了对于任何给定的被试，属于同一个次级分组（例如，来自同一次试验的所有数据）的所有数据点完全位于该被试的训练部分或验证部分。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_subject_X.csv`, `val_subject_X.csv`），则使用这些文件；否则，将根据每个被试的数据和 `group_by` 标准生成新的切分，并保存。`split` 方法会为每个被试（或特定的一个被试）产出这些训练/验证数据对。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于在*每个被试内部*对数据进行分组以便切分的列名（例如 'trial_id', 'session_id'）。
- **val_size** (`float`): 在每个被试的数据内，要包含在其各自验证集中的唯一组（由 `group_by` 定义）的比例。应介于0.0和1.0之间。默认为 `0.2`。
- **shuffle** (`bool`): 在切分每个被试内部的组（由 `group_by` 定义）之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[int, None]`): 控制每个被试内部分组的打乱。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每个被试切分文件（例如 `train_subject_X.csv`, `val_subject_X.csv`）的目录路径。这些文件将定义*每个被试内部*各组的训练/验证切分。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含一个 'subject_id' 列
# 以及由 'group_by' 指定的列 (例如, 'trial_id')。

# 示例：对于每个被试，切分其试验 ('group_by')，使得
# 该被试20%的试验进入该被试的验证集。
per_subject_group_splitter = HoldOutPerSubjectCross(
    group_by='trial_id',   # 在每个被试内部按试验ID分组数据
    val_size=0.20,
    shuffle=True,
    random_state=42,
    split_path='存放每个被试分组切分文件的路径' # 替换为实际路径
)

# 遍历所有被试的切分：
# 'train_set' 和 'val_set' 将是 BaseDataset 实例，其 '.info' 属性
# 特定于某个被试的训练/验证部分，其中切分是通过将试验保持在一起完成的。
# for train_set, val_set, test_set_passthrough in per_subject_group_splitter.split(dataset):
#     # 这个循环会为每个被试运行。
#     # current_subject_id = train_set.info['subject_id'].unique()[0]
#     # print(f"正在处理被试: {current_subject_id}")
#     # print(f"  训练集试验: {sorted(train_set.info['trial_id'].unique())}")
#     # print(f"  验证集试验: {sorted(val_set.info['trial_id'].unique())}")
#     pass

# 示例：获取特定被试的切分 (例如, 被试 'S01')
specific_subject_id = 'S01' # 替换为实际的被试ID字符串
for train_set_s01, val_set_s01, test_set_s01 in per_subject_group_splitter.split(dataset, subject=specific_subject_id):
    # 这个循环将仅为被试 'S01' 运行。
    # print(f"正在处理特定被试: {specific_subject_id}")
    # print(f"  训练集试验: {sorted(train_set_s01.info['trial_id'].unique())}")
    # print(f"  验证集试验: {sorted(val_set_s01.info['trial_id'].unique())}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
## K Fold

### KFold

~~~python
class KFold(BaseSplit):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现K折交叉验证。此类使用 `sklearn.model_selection.KFold` 将数据集的 `info` DataFrame划分为 `n_splits` 折。对于每一折，一部分用作验证集，其余部分用作训练集。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_fold_0.csv`, `val_fold_0.csv`），则使用这些切分；否则，将根据数据集的 `info` DataFrame为每一折生成新的切分，并保存到 `split_path`。`split` 方法随后会为每一折产出这些训练/验证数据对。

**参数**

- **n_splits** (`int`): 折数。必须至少为2。默认为 `5`。
- **shuffle** (`bool`): 在将数据分成批次之前是否打乱数据。请注意，每个切分内部的样本不会被打乱。默认为 `False`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，`random_state` 会影响索引的排序，从而控制每一折的随机性。传递一个整数以确保多次函数调用之间的输出可复现。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每折切分文件（例如 `train_fold_X.csv`, `val_fold_X.csv`）的目录路径。如果路径不存在或未包含所有折的正确命名文件，则会生成新的切分并保存。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame。

# 示例：执行5折交叉验证。
k_fold_splitter = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='存放K折切分文件的路径' # 替换为实际的目录路径
)

# 'split' 方法是一个生成器，为每一折产出一个训练/验证数据对。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已为当前折更新。
# 如果没有向 .split() 方法传递 test_dataset，则 'test_set_passthrough' 将为 None。
fold_counter = 0
for train_set, val_set, test_set_passthrough in k_fold_splitter.split(dataset):
    fold_counter += 1
    # print(f"正在处理第 {fold_counter}/{k_fold_splitter.n_splits} 折")
    # print(f"  训练集大小: {len(train_set.info)}")
    # print(f"  验证集大小: {len(val_set.info)}")
    # 在这一折中使用 train_set 和 val_set 进行训练和验证。
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### KFoldCross

~~~python
class KFoldCross(BaseSplit):
    def __init__(
        self,
        group_by: str,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

通过按 `group_by` 列（例如 'subject_id', 'session_id'）标识的唯一组进行切分，来实现K折交叉验证。这确保了属于同一组的所有数据点在所有K个切分中都被分配到同一折，用于训练或验证。当样本不独立时（例如，来自同一被试的多次试验），这对于防止数据泄漏特别有用。组ID的实际K折切分是使用 `sklearn.model_selection.KFold` 执行的。如果 `split_path` 已提供并且包含每折的现有切分文件，则使用这些文件；否则，将生成新的切分并保存。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于对数据进行分组的列名。K折切分将对此列中的唯一值（组）进行。
- **n_splits** (`int`): 折数。必须至少为2。默认为 `5`。
- **shuffle** (`bool`): 在将唯一组切分为批次（折）之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，`random_state` 会影响组的排序，从而控制每一折的随机性。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每折切分文件（例如 `train_fold_X.csv`, `val_fold_X.csv`）的目录路径。这些文件将定义每折中组的训练/验证切分。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'group_by' 指定的列 (例如, 'subject_id')。

# 示例：执行5折交叉验证，按 'subject_id' 分组。
# 这意味着被试将被分配到不同的折，而不是单个样本。
k_fold_group_splitter = KFoldCross(
    group_by='subject_id', # 按被试ID对数据进行分组以进行K折切分
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='存放分组K折切分文件的路径' # 替换为实际的目录路径
)

# 'split' 方法是一个生成器，为每一折产出一个训练/验证数据对。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已为当前折更新，
# 包含所选组的所有数据。
fold_counter = 0
for train_set, val_set, test_set_passthrough in k_fold_group_splitter.split(dataset):
    fold_counter += 1
    # print(f"正在处理第 {fold_counter}/{k_fold_group_splitter.n_splits} 折")
    # print(f"  训练集组 (例如，被试): {sorted(train_set.info['subject_id'].unique())}")
    # print(f"  验证集组 (例如，被试): {sorted(val_set.info['subject_id'].unique())}")
    # 在这一折中使用 train_set 和 val_set 进行训练和验证。
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### KFoldGroupby

~~~python
class KFoldGroupby(BaseSplit):
    def __init__(
        self,
        group_by: str,
        base_group: str = 'subject_id',
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种具有层级分组结构的K折交叉验证。该过程首先遍历每个唯一的 `base_group`（例如 'subject_id'）。在每个 `base_group` 内部，它再根据 `group_by` 列（例如，该被试的 'trial_id' 或 'session_id'）识别唯一的次级组。对于这些次级组中的每一个，K折切分将应用于其包含的数据样本。最后，来自所有次级组（跨所有基本组）的相应训练集和验证集将被聚合起来，形成最终的K个折。这确保了每个次级组内的数据会分布到K个折中。如果 `split_path` 已提供并且包含每折的现有切分文件，则使用这些文件；否则，将生成新的切分并保存。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于次级分组的列名（例如 'trial_id', 'session_id'）。K折切分将应用于这些组*内部*的数据样本。
- **base_group** (`str`): 数据集 `info` DataFrame 中用于主要、较粗粒度分组的列名（例如 'subject_id'）。切分过程将针对此基本组中的每个唯一值进行迭代。默认为 `'subject_id'`。
- **n_splits** (`int`): 应用于每个次级组内部数据样本的K折切分的折数。必须至少为2。默认为 `5`。
- **shuffle** (`bool`): 在将每个次级组内的数据样本切分为K折之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，`random_state` 会影响每个次级组内样本的排序，从而控制每一折的随机性。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每折切分文件（例如 `train_fold_X.csv`, `val_fold_X.csv`）的目录路径。这些文件将包含聚合后的训练/验证切分。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'base_group' (例如 'subject_id') 
# 和 'group_by' (例如 'trial_id') 指定的列。

# 示例：对于每个被试 ('base_group')，以及该被试的每次试验 ('group_by')，
# 对该次试验的样本应用5折交叉验证。最终的K个折是这些切分的聚合。
hierarchical_kfold_splitter = KFoldGroupby(
    group_by='trial_id',      # 在每次试验内部进行K折切分
    base_group='subject_id',  # 遍历所有被试
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='存放层级K折切分文件的路径' # 替换为实际路径
)

# 'split' 方法是一个生成器，为K个聚合折中的每一折产出一个训练/验证数据对。
# 'train_set' 和 'val_set' 将是新的 BaseDataset 实例，其 '.info' 属性已为当前聚合折更新。
fold_counter = 0
for train_set, val_set, test_set_passthrough in hierarchical_kfold_splitter.split(dataset):
    fold_counter += 1
    # print(f"正在处理聚合的第 {fold_counter}/{hierarchical_kfold_splitter.n_splits} 折")
    # print(f"  训练集大小: {len(train_set.info)}")
    # print(f"  验证集大小: {len(val_set.info)}")
    # 在这一聚合折中使用 train_set 和 val_set 进行训练和验证。
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### KFoldPerSubject

~~~python
class KFoldPerSubject(BaseSplit):
    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[int, None] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

在每个被试的基础上独立实现K折交叉验证。对于数据集中 `info` DataFrame 里由 'subject_id' 标识的每个被试，其个体数据将使用 `sklearn.model_selection.KFold` 被切分为 `n_splits` 折。这意味着对于每个被试，他们的数据样本将被用来创建K个针对该被试的不同训练/验证切分。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_subject_X_fold_Y.csv`），则使用这些文件；否则，将为每个被试的每一折生成新的切分，并保存。`split` 方法会为所有被试（或特定请求的被试）的所有折产出这些训练/验证数据对。

**参数**

- **n_splits** (`int`): 为每个被试的数据创建的折数。必须至少为2。默认为 `5`。
- **shuffle** (`bool`): 在将每个被试的数据切分为批次（折）之前是否对其进行打乱。请注意，此参数不会进一步打乱每个切分内部的样本。默认为 `False`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，`random_state` 会影响每个被试数据的索引排序，从而控制该被试每一折的随机性。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每个被试每折切分文件（例如 `train_subject_X_fold_Y.csv`, `val_subject_X_fold_Y.csv`）的目录路径。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含一个 'subject_id' 列。

# 示例：对于每个被试，对其数据执行5折交叉验证。
per_subject_kfold_splitter = KFoldPerSubject(
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='存放每个被试K折切分文件的路径' # 替换为实际的目录路径
)

# 遍历所有被试和所有折的切分：
# 'train_set' 和 'val_set' 将是 BaseDataset 实例，其 '.info' 属性
# 特定于某个被试在某一折的训练/验证部分。
# for train_set, val_set, test_set_passthrough in per_subject_kfold_splitter.split(dataset):
#     # 这个循环会对每个被试运行 n_splits 次。
#     # current_subject_id = train_set.info['subject_id'].unique()[0] # 获取当前被试的示例
#     # print(f"正在处理被试: {current_subject_id}, 折数: ...") # 如果需要，此处需要一种方式获取 fold_id
#     # print(f"  训练集大小: {len(train_set.info)}, 验证集大小: {len(val_set.info)}")
#     pass

# 示例：获取特定被试的所有K折切分 (例如, 被试 'S01')
specific_subject_id = 'S01' # 替换为实际的被试ID字符串
fold_count_s01 = 0
for train_set_s01, val_set_s01, test_s01 in per_subject_kfold_splitter.split(dataset, subject=specific_subject_id):
    fold_count_s01 += 1
    # 这个循环将仅为被试 'S01' 运行 n_splits 次。
    # print(f"正在处理特定被试: {specific_subject_id}, 折数: {fold_count_s01}")
    # print(f"  训练集大小: {len(train_set_s01.info)}, 验证集大小: {len(val_set_s01.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### KFoldPerSubjectCross

~~~python
class KFoldPerSubjectCross(BaseSplit):
    def __init__(
        self,
        group_by: str,
        n_splits: int = 5,
        shuffle: bool = False,
        random_state: Union[None, int] = None,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

在每个被试的基础上独立实现K折交叉验证，其中*在每个被试的数据内部*的切分是基于指定的 `group_by` 列（例如 'trial_id', 'session_id'）进行分组。对于每个被试，其唯一的（次级）组（例如试验）将被划分为 `n_splits` 折。这确保了对于特定被试，属于某个特定组（例如一次试验）的所有数据点在该被试的交叉验证的每一折中，都完整地保留在训练集或验证集里。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_subject_X_fold_Y.csv`），则使用这些文件；否则，将生成新的切分并保存。`split` 方法会为所有被试（或特定请求的被试）的所有折产出这些训练/验证数据对。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于在*每个被试内部*对数据进行分组的列名。K折切分将对这些唯一的被试内部分组进行。
- **n_splits** (`int`): 为每个被试数据内部的组创建的折数。必须至少为2。默认为 `5`。
- **shuffle** (`bool`): 在将每个被试内部的唯一组（由 `group_by` 定义）切分为折之前是否对其进行打乱。默认为 `False`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，`random_state` 会影响每个被试内部分组的排序，从而控制该被试每一折的随机性。传递一个整数以确保可复现的输出。默认为 `None`。
- **split_path** (`Union[None, str]`): 用于存储每个被试每折切分文件（例如 `train_subject_X_fold_Y.csv`, `val_subject_X_fold_Y.csv`）的目录路径。这些文件将定义*每个被试内部*各组在每一折的训练/验证切分。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含一个 'subject_id' 列
# 以及由 'group_by' 指定的列 (例如, 'trial_id')。

# 示例：对于每个被试，对其试验数据进行5折交叉验证。
# 这意味着对于每个被试，其试验（而不是单个样本）被分配到不同的折中。
per_subject_trial_kfold_splitter = KFoldPerSubjectCross(
    group_by='trial_id', # 在每个被试内部按试验ID分组进行K折切分
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='存放每个被试试验K折切分文件的路径' # 替换为实际路径
)

# 遍历所有被试和所有折的切分：
# 'train_set' 和 'val_set' 将是 BaseDataset 实例，其 '.info' 属性
# 特定于某个被试在某一折的训练/验证部分，其中试验是分配到训练/验证集的基本单元。
# for train_set, val_set, test_set_passthrough in per_subject_trial_kfold_splitter.split(dataset):
#     # 这个循环会对每个被试运行 n_splits 次。
#     # current_subject_id = train_set.info['subject_id'].unique()[0]
#     # print(f"正在处理被试: {current_subject_id}, 折数: ... ") # 如果需要，此处需要一种方式获取 fold_id
#     # print(f"  训练集试验: {sorted(train_set.info['trial_id'].unique())}")
#     # print(f"  验证集试验: {sorted(val_set.info['trial_id'].unique())}")
#     pass

# 示例：获取特定被试的所有K折切分 (例如, 被试 'S01')
specific_subject_id = 'S01' # 替换为实际的被试ID字符串
fold_count_s01 = 0
for train_set_s01, val_set_s01, test_s01 in per_subject_trial_kfold_splitter.split(dataset, subject=specific_subject_id):
    fold_count_s01 += 1
    # 这个循环将仅为被试 'S01' 运行 n_splits 次。
    # print(f"正在处理特定被试: {specific_subject_id}, 折数: {fold_count_s01}")
    # print(f"  被试S01的训练集试验: {sorted(train_set_s01.info['trial_id'].unique())}")
    # print(f"  被试S01的验证集试验: {sorted(val_set_s01.info['trial_id'].unique())}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
## Leave One Out

### LeaveOneOut

~~~python
class LeaveOneOut(BaseSplit):
    def __init__(
        self, 
        group_by: str,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现留一组交叉验证（Leave-One-Group-Out）。在此策略中，如果有N个唯一组（例如，由 `group_by` 列定义的被试、会话或试验），则执行N次迭代。在每次迭代 `i` 中，第i个组被作为验证集留出，所有其他N-1个组构成训练集。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_groupX_Y.csv`, `val_groupX_Y.csv`），则使用这些文件；否则，将生成新的切分并保存。`split` 方法随后可以为所有组或为特定请求留出的组产出这些训练/验证数据对。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于分组的列名。此列中的每个唯一值将形成一个组，在一次迭代中被留出。
- **split_path** (`Union[None, str]`): 用于存储切分文件（根据 `group_by` 列和留出的组命名）的目录路径。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'group_by' 指定的列 (例如, 'subject_id')。

# 示例：执行留一被试交叉验证。
leave_one_subject_out_splitter = LeaveOneOut(
    group_by='subject_id', # 每个被试将依次作为验证集
    split_path='存放留一法切分文件的路径' # 替换为实际的目录路径
)

# 遍历切分，每次留出一个被试：
# 'train_set' 包含所有其他被试，'val_set' 包含被留出的被试。
# 如果没有向 .split() 方法传递 test_dataset，则 'test_set_passthrough' 将为 None。
# for train_set, val_set, test_set_passthrough in leave_one_subject_out_splitter.split(dataset):
#     # 这个循环会对数据集中的每个唯一 subject_id 运行。
#     # left_out_subject = val_set.info['subject_id'].unique()[0]
#     # print(f"留出被试: {left_out_subject}")
#     # print(f"  训练集大小: {len(train_set.info)}, 验证集大小: {len(val_set.info)}")
#     pass

# 示例：获取特定被试（例如 'S03'）被留出作为验证集的切分
specific_group_to_leave_out = 'S03' # 替换为实际的组ID (例如，被试ID)
for train_set_loo, val_set_loo, test_set_loo in leave_one_subject_out_splitter.split(dataset, group=specific_group_to_leave_out):
    # 这个循环将仅为指定的组运行一次。
    # print(f"特定组 '{specific_group_to_leave_out}' 被留出作为验证集。")
    # print(f"  训练集大小: {len(train_set_loo.info)}, 验证集大小: {len(val_set_loo.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### LeaveOneOutEAndT

~~~python
class LeaveOneOutEAndT(BaseSplit):
    def __init__(
        self, 
        group_by: str,
        split_path: Union[None, str] = None,
        **kwargs
    ) -> None:
~~~

实现一种留一法风格的交叉验证，为每次迭代生成训练集、验证集和测试集。在此策略中，对于每个被指定为当次迭代的 `test_group` 的唯一组（由 `group_by` 定义，例如 'subject_id'）：

1. `test_group` 本身构成测试集。
2. 紧邻 `test_group` 之前（从所有唯一组的排序列表中循环选择）的组被选为验证集 (`val_group`)。
3. 所有其他剩余的组构成训练集。 此过程会重复进行，每个组轮流作为 `test_group`。如果 `split_path` 已提供并且包含现有的切分文件（例如 `train_groupX_Y.csv`, `val_groupX_Y.csv`, `test_groupX_Y.csv`，其中Y是该次迭代的 `test_group`），则使用这些文件；否则，将生成新的切分并保存。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于分组的列名。此列中的每个唯一值将定义一个组。
- **split_path** (`Union[None, str]`): 用于存储切分文件（根据 `group_by` 列和该次迭代中指定为 `test_group` 的组来命名）的目录路径。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'group_by' 指定的列 (例如, 'session_id')。

# 示例：执行留一会话法，同时选择一个验证会话。
loo_et_splitter = LeaveOneOutEAndT(
    group_by='session_id', # 每个会话将依次作为测试集
    split_path='存放LOO-ET切分文件的路径' # 替换为实际的目录路径
)

# 遍历切分：
# 在每次迭代中：
# - 'test_s' 包含当前作为测试集留出的会话。
# - 'val_s' 包含（循环地）在 test_s 之前的会话作为验证集。
# - 'train_s' 包含所有其他会话。
# for train_s, val_s, test_s in loo_et_splitter.split(dataset):
#     # 这个循环会对数据集中的每个唯一 session_id 运行。
#     # current_test_session = test_s.info['session_id'].unique()[0]
#     # current_val_session = val_s.info['session_id'].unique()[0]
#     # print(f"测试会话: {current_test_session}, 验证会话: {current_val_session}")
#     # print(f"  训练集: {len(train_s.info)}, 验证集: {len(val_s.info)}, 测试集: {len(test_s.info)}")
#     pass

# 示例：获取 'Session3' 作为测试组的特定切分。
# 验证组将是 'Session3' 之前的那个组。
specific_test_group = 'Session3' # 替换为实际的组ID
for train_specific, val_specific, test_specific in loo_et_splitter.split(dataset, group=specific_test_group):
    # 这个循环将仅为指定的测试组运行一次。
    # print(f"测试组: {specific_test_group}")
    # print(f"  训练集: {len(train_specific.info)}, 验证集: {len(val_specific.info)}, 测试集: {len(test_specific.info)}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
### LosoRotatingCrossSplit

~~~python
class LosoRotatingCrossSplit(BaseSplit):
    def __init__(
        self,
        group_by: str,
        split_path: Union[None, str] = None,
        n_coarse_splits: int = 4,
        num_val_groups_in_rotation: int = 2,
        num_test_groups_in_rotation: int = 1,
        shuffle: bool = True,
        random_state: Union[None, int] = None,
        **kwargs
    ) -> None:
~~~

实现一种类似“留一会话输出”（LOSO）的轮转交叉验证策略。该过程包含两个主要阶段：

1. **粗略K折切分**：首先将唯一的组（由 `group_by` 列定义，例如 'session_id' 或 'subject_id'）划分为 `n_coarse_splits` 折。在粗略切分的每次迭代中，一折成为“剩余集”，其他折构成训练集的初始部分。

2. 轮转验证/测试集

   ：对于从粗略切分中获得的每个“剩余集”，应用轮转机制。在轮转的每一步中：

   - 从（循环移位的）“剩余集”的末尾指定数量的组（`num_test_groups_in_rotation`）被指定为当前生成折的测试集。
   - 紧邻测试集组之前的指定数量的组（`num_val_groups_in_rotation`）被指定为验证集。
   - “剩余集”中的其余组与来自粗略切分的初始训练组合并，形成当前生成折的最终训练集。 此过程会生成多个训练/验证/测试切分。如果 `split_path` 已提供并且包含现有的、正确命名的切分文件（例如 `train_fold_X.csv`），则使用这些文件；否则，将生成新的切分并保存。

**参数**

- **group_by** (`str`): 数据集 `info` DataFrame 中用于定义将要被切分和轮转的唯一组的列名（例如 'session_id', 'subject_id'）。
- **split_path** (`Union[None, str]`): 用于存储切分定义文件（例如 `train_fold_X.csv`, `val_fold_X.csv`, `test_fold_X.csv`）的目录路径。默认为 `None`。
- **n_coarse_splits** (`int`): 初始粗略K折切分唯一组时的折数。默认为 `4`。
- **num_val_groups_in_rotation** (`int`): 在每次轮转中，从粗略切分的“剩余集”中用于验证集的组的数量。默认为 `2`。
- **num_test_groups_in_rotation** (`int`): 在每次轮转中，从粗略切分的“剩余集”中用于测试集的组的数量。必须大于0。默认为 `1`。
- **shuffle** (`bool`): 在初始粗略K折切分之前是否打乱唯一组。默认为 `True`。
- **random_state** (`Union[None, int]`): 当 `shuffle` 为 `True` 时，此种子用于粗略K折打乱，确保可复现性。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'dataset' 是一个已存在的 BaseDataset 实例。
# 'dataset.info' 应为一个 pandas DataFrame，并且必须包含由 'group_by' 指定的列 (例如, 'session_id')。

# 示例：实现轮转交叉切分，按 'session_id' 分组。
# 粗略地将session划分为4折。对于每组粗略划分中“留出”的session部分，
# 对其进行轮转，使得1个session用于测试，其前的2个session用于验证。
rotating_splitter = LosoRotatingCrossSplit(
    group_by='session_id',
    n_coarse_splits=4,
    num_val_groups_in_rotation=2,
    num_test_groups_in_rotation=1,
    shuffle=True,
    random_state=42,
    split_path='存放LOSO轮转切分文件的路径' # 替换为实际路径
)

# 'split' 方法是一个生成器，为每个生成的折产出一个训练/验证/测试三元组。
# 产出的折数将是 n_coarse_splits * (每个“剩余集”中的组数)。
fold_counter = 0
for train_set, val_set, test_set in rotating_splitter.split(dataset):
    fold_counter += 1
    # print(f"正在处理生成的第 {fold_counter} 折")
    # print(f"  训练集组 (例如, sessions): {sorted(train_set.info['session_id'].unique())}")
    # print(f"  验证集组: {sorted(val_set.info['session_id'].unique())}")
    # print(f"  测试集组: {sorted(test_set.info['session_id'].unique())}")
    # 确保此折的训练、验证、测试组之间没有重叠。
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
## NoSplit

~~~python
class NoSplit(BaseSplit):
    def __init__(
        self, 
        split_path: str = None,
        **kwargs
    ) -> None:
~~~

此类代表一种不执行实际数据集切分的策略。它直接将提供的 `dataset`、`val_dataset` 和 `test_dataset` 作为单个“切分”产出。当预切分的数据集已经可用，并且需要在期望切分生成机制的框架内使用时，或者当需要在完整训练集上进行评估时（尽管通常验证集仍然是不同的），这种策略非常有用。`split_path` 参数被继承但此类不主动使用，因为不生成或读取切分文件。

**参数**

- **split_path** (`str`, optional): 指向切分数据集信息的路径。此参数继承自 `BaseSplit`，但 `NoSplit` 类不使用它，因为不发生实际的切分或加载/保存切分文件。默认为 `None`。
- ***\*kwargs**: 传递给父类 `BaseSplit` 的其他关键字参数。

**使用样例**

~~~python
# 假设 'train_dataset_predefined', 'validation_dataset_predefined', 
# 和 'test_dataset_predefined' 是已存在的 BaseDataset 实例。

# 1. 实例化 NoSplit 策略。
#    'split_path' 不被 NoSplit 使用，但它是 BaseSplit 签名的一部分。
no_split_strategy = NoSplit(split_path=None)

# 2. “切分”数据集。
#    这将把原始数据集作为一个元组直接返回。
for train_set, val_set, test_set in no_split_strategy.split(
    dataset=train_dataset_predefined, 
    val_dataset=validation_dataset_predefined, 
    test_dataset=test_dataset_predefined
):
    # 'train_set' 将与 'train_dataset_predefined' 相同。
    # 'val_set' 将与 'validation_dataset_predefined' 相同。
    # 'test_set' 将与 'test_dataset_predefined' 相同。

    # 示例：打印关于产出数据集的基本信息
    # if train_set:
    #     print(f"产出的训练集 .info 长度: {len(train_set.info) if hasattr(train_set, 'info') else 'N/A'}")
    # if val_set:
    #     print(f"产出的验证集 .info 长度: {len(val_set.info) if hasattr(val_set, 'info') else 'N/A'}")
    # if test_set:
    #     print(f"产出的测试集 .info 长度: {len(test_set.info) if hasattr(test_set, 'info') else 'N/A'}")
    pass
~~~
[`返回顶部`](#tyeedatasetsplit)
