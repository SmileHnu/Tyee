# tyee.dataset.split

`tyee.dataset.split` module provides a variety of dataset division methods for users to choose from different methods.

| Class Name                 | Function/Purpose                                             |
| -------------------------- | ------------------------------------------------------------ |
| [`BaseSplit`](#basesplit) | Base class for all splitting strategies, defining a common interface. |
| [`HoldOut`](#holdout)                | Performs a simple, one-time train/validation split of the entire dataset. |
| [`HoldOutCross`](#holdoutcross)           | Splits the entire dataset into train/validation sets based on groups (e.g., 'subject_id'), ensuring no group is split across sets. |
| [`HoldOutGroupby`](#holdoutgroupby)         | Hierarchical hold-out: first groups by `base_group` (e.g., 'subject_id'), then within each, splits unique units of `group_by` (e.g., 'trial_id') into train/val, and aggregates results. |
| [`HoldOutPerSubject`](#holdoutpersubject)      | Independently splits data for *each subject* into a training and validation set. The `split` method iterates through these per-subject splits. |
| [`HoldOutPerSubjectET`](#holdoutpersubjectet)    | Independently splits data for *each subject* into training, validation, *and test* sets. The `split` method iterates through these per-subject splits. |
| [`HoldOutPerSubjectCross`](#holdoutpersubjectcross) | For *each subject*, splits their internal groups (defined by `group_by`, e.g., trials) into train and validation portions for that subject. Iterates through subjects. |
| [`KFold`](#kfold)                  | Standard K-fold cross-validation on the samples of the entire dataset. |
| [`KFoldCross`](#kfoldcross)             | K-fold cross-validation where groups (defined by `group_by`, e.g., subject IDs) are distributed into K folds, ensuring group integrity. |
| [`KFoldGroupby`](#kfoldgroupby)           | Hierarchical K-fold: groups by `base_group`, then for each `group_by` unit within that, applies K-fold to its *samples*. Final K folds are aggregations. |
| [`KFoldPerSubject`](#kfoldpersubject)        | Applies K-fold cross-validation independently to the data samples of *each subject*. The `split` method iterates through all folds for all subjects. |
| [`KFoldPerSubjectCross`](#kfoldpersubjectcross)   | For *each subject*, applies K-fold cross-validation to their internal *groups* (defined by `group_by`, e.g., trials). Iterates through subjects and their K group-based folds. |
| [`LeaveOneOut`](#leaveoneout)            | Leave-One-Group-Out: each group (defined by `group_by`) is used as the validation set once, with all others for training. |
| [`LeaveOneOutEAndT`](#leaveoneouteandt)       | Leave-One-Out variant: for each group designated as test, another (preceding) group is validation, and the rest is train. |
| [`LosoRotatingCrossSplit`](#losorotatingcrosssplit) | Complex rotating split: performs a coarse K-fold on groups, then rotates subsets of the "rest" set for validation and testing, generating multiple train/val/test splits. |
| [`NoSplit`](#nosplit)                | Does not perform any actual data splitting; yields back the input training, validation (optional), and test (optional) datasets as is. Suitable for pre-split data. |



## BaseSplit

~~~python
class BaseSplit:
    def __init__(
        self,
        split_path: str,
        **kwargs
    ) -> None:
~~~

This is the base class for defining dataset splitting strategies. It provides a common structure for initializing the split mechanism with a path and includes a method to check the validity of this path. Subclasses are expected to implement the actual `split` logic.

**Parameters**

- **split_path** (`str`): The path to a directory where split information (e.g., CSV files defining train/validation/test splits) is located or will be stored.
- ***\*kwargs**: Additional keyword arguments that might be used by subclasses.

**Methods**

~~~python
def check_split_path(self) -> bool:
~~~

Checks if the `split_path` directory exists and contains at least one CSV file. It logs information about its findings.

*Returns:*

- `bool`: `True` if the path exists and contains at least one CSV file, `False` otherwise.

~~~python
def split(
    self, 
    dataset: BaseDataset,
    val_dataset: BaseDataset = None,
    test_dataset: BaseDataset = None,
) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
~~~

This method is intended to be overridden by subclasses to implement specific dataset splitting logic. It should define how a given `dataset` (and optionally `val_dataset` and `test_dataset`) is divided into training, validation, and testing sets.

*Args:*

- **dataset** (`BaseDataset`): The primary dataset to be split.
- **val_dataset** (`BaseDataset`, optional): An optional, separate validation dataset.
- **test_dataset** (`BaseDataset`, optional): An optional, separate test dataset.

*Yields:*

- `Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]`: A generator that should yield tuples, typically representing `(train_dataset, validation_dataset, test_dataset)`.

*Raises:*

- `NotImplementedError`: This base method must be overridden by subclasses.

`BaseSplit` is an abstract base class and is not typically instantiated directly.

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out validation strategy, splitting a dataset into a single training set and a validation set. If `split_path` is provided and contains existing `train.csv` and `val.csv` files, those splits are used; otherwise, a new split is generated based on the dataset's `info` DataFrame and saved to `split_path`. An optional test set can be passed through but is not actively split by this method.

**Parameters**

- **val_size** (`float`): The proportion of the dataset to include in the validation split. Should be between 0.0 and 1.0. Defaults to `0.2`.
- **shuffle** (`bool`): Whether to shuffle the data before splitting. Defaults to `False`.
- **stratify** (`str`, optional): If not `None`, data is split in a stratified fashion, using this string as the column name in the dataset's `info` DataFrame for stratification. Defaults to `None`.
- **random_state** (`Union[int, None]`): Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where `train.csv` and `val.csv` (defining the splits) are stored or will be saved. If the path does not exist or does not contain these files, new splits are generated and saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame.
# If stratify='label' is used, 'dataset.info' should have a 'label' column.

hold_out_splitter = HoldOut(
    val_size=0.25,
    shuffle=True,
    stratify='label', # Assumes 'label' column in dataset.info
    random_state=42,
    split_path='PATH_TO_YOUR_SPLIT_FILES' # Replace with the actual directory path
)

# The 'split' method is a generator.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated to reflect the respective splits.
# 'test_set_passthrough' will be None if no test_dataset was passed to .split().
for train_set, val_set, test_set_passthrough in hold_out_splitter.split(dataset):
    # Use train_set and val_set for training and validation.
    # Example: print(f"Train set size: {len(train_set.info)}, Validation set size: {len(val_set.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out cross-validation strategy where the dataset is split into training and validation sets based on a specified grouping column (e.g., 'subject_id', 'session_id'). This ensures that all data points belonging to the same group (e.g., all trials from one subject) are entirely within either the training set or the validation set, preventing data leakage between splits from the same group. If `split_path` is provided and contains existing `train.csv` and `val.csv` files, those splits are used; otherwise, new splits are generated based on the unique group IDs in the dataset's `info` DataFrame and saved to `split_path`.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame to use for grouping the data before splitting (e.g., 'trial_id', 'session_id', 'subject_id').
- **val_size** (`float`): The proportion of the unique groups to include in the validation split. This should be between 0.0 and 1.0. Defaults to `0.2`.
- **shuffle** (`bool`): Whether to shuffle the unique group IDs before splitting them into training and validation groups. Defaults to `False`.
- **random_state** (`Union[None, int]`): Controls the shuffling of group IDs. Pass an int for reproducible splits across multiple function calls. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where `train.csv` and `val.csv` (defining the splits based on the original dataset's `info`) are stored or will be saved. If the path does not exist or does not contain these files, new splits are generated and saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the column specified by 'group_by' (e.g., 'subject_id').

# Example: Split by 'subject_id', with 20% of subjects in the validation set.
hold_out_cross_splitter = HoldOutCross(
    group_by='subject_id', # Group data by subject
    val_size=0.2,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_GROUPED_SPLIT_FILES' # Replace with actual directory path
)

# The 'split' method is a generator.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated to reflect the group-wise splits.
# 'test_set_passthrough' will be None if no test_dataset was passed to .split().
for train_set, val_set, test_set_passthrough in hold_out_cross_splitter.split(dataset):
    # Use train_set and val_set for training and validation.
    # All data from any given subject_id will be entirely in train_set or val_set.
    # Example: print(f"Train set info length: {len(train_set.info)}, Validation set info length: {len(val_set.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out validation strategy with hierarchical grouping. The dataset is split into training and validation sets by first considering a `base_group` (e.g., 'subject_id'), and then, within each base group, splitting a secondary `group_by` (e.g., 'trial_id' or 'session_id'). This ensures that all data points belonging to the same secondary group within a base group are entirely in either the training or validation set. The splits are performed independently for each base group, and then the resulting training and validation sets are aggregated across all base groups. If `split_path` is provided and contains existing `train.csv` and `val.csv` files, those are used; otherwise, new splits are generated and saved.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame used for the secondary level of grouping (e.g., 'trial_id', 'session_id'). The split into train/validation will occur at this group level, within each `base_group`.
- **base_group** (`str`): The column name in the dataset's `info` DataFrame used for the primary, coarser level of grouping (e.g., 'subject_id'). The splitting process is iterated for each unique value in this base group. Defaults to `"subject_id"`.
- **val_size** (`float`): The proportion of the secondary groups (defined by `group_by`) within each `base_group` to include in the validation split. Should be between 0.0 and 1.0. Defaults to `0.2`.
- **shuffle** (`bool`): Whether to shuffle the secondary groups (within each base group) before splitting them. Defaults to `False`.
- **random_state** (`Union[int, None]`): Controls the shuffling of secondary groups. Pass an int for reproducible splits. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where `train.csv` and `val.csv` (defining the aggregated splits) are stored or will be saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the columns specified by 'base_group' (e.g., 'subject_id') and 
# 'group_by' (e.g., 'session_id').

# Example: For each subject ('base_group'), split their sessions ('group_by')
# such that 20% of sessions from each subject go into the validation set.
hierarchical_hold_out_splitter = HoldOutGroupby(
    group_by='session_id',    # Split sessions within each subject
    base_group='subject_id',  # Iterate over subjects
    val_size=0.20,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_HIERARCHICAL_SPLIT_FILES' # Replace with actual path
)

# The 'split' method is a generator.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated to reflect the aggregated hierarchical splits.
for train_set, val_set, test_set_passthrough in hierarchical_hold_out_splitter.split(dataset):
    # Use train_set and val_set for training and validation.
    # All data from any given session_id (within a subject_id) will be 
    # entirely in train_set or val_set for that subject's portion of the split.
    # Example: print(f"Aggregated Train set size: {len(train_set.info)}, Aggregated Validation set size: {len(val_set.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out validation strategy where the data for *each subject* is independently split into a training and a validation set. This means that for every subject in the dataset, a portion of their data (`val_size`) is set aside for validation, and the rest for training. If `split_path` is provided and contains existing split files (e.g., `train_subject_X.csv`, `val_subject_X.csv`), those splits are used for the respective subjects; otherwise, new splits are generated for each subject based on their data in the dataset's `info` DataFrame and saved to `split_path`. The `split` method can then yield these train/validation pairs for all subjects or for a specifically requested subject.

**Parameters**

- **val_size** (`float`): The proportion of each subject's data to include in their respective validation split. Should be between 0.0 and 1.0. Defaults to `0.2`.
- **shuffle** (`bool`): Whether to shuffle each subject's data before splitting it. Defaults to `False`.
- **random_state** (`Union[int, None]`): Controls the shuffling applied to each subject's data. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-subject split files (`train_subject_X.csv`, `val_subject_X.csv`) are stored or will be saved. If `None`, the behavior might depend on `BaseSplit` or internal handling, but typically this path is needed for saving/loading. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# a 'subject_id' column.

# Example: For each subject, set aside 25% of their data for validation.
per_subject_splitter = HoldOutPerSubject(
    val_size=0.25,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_PER_SUBJECT_SPLIT_FILES' # Replace with actual directory path
)

# Iterate through splits for all subjects:
# 'train_set' and 'val_set' will be BaseDataset instances with their '.info'
# attributes specific to one subject's train/val portion.
# 'test_set_passthrough' is typically None or a general test set passed along.
# for train_set, val_set, test_set_passthrough in per_subject_splitter.split(dataset):
#     # This loop will run for each subject found based on the split files or dataset.info.
#     # Example: print(f"Processing subject: {train_set.info['subject_id'].iloc[0]}")
#     # print(f"  Train set size: {len(train_set.info)}, Validation set size: {len(val_set.info)}")
#     pass

# Example: Get split for a specific subject (e.g., subject 'S01')
# Ensure 'S01' is a valid subject ID present in the dataset or split files.
specific_subject_id = 'S01' # Replace with an actual subject ID string
for train_set_s01, val_set_s01, test_set_s01 in per_subject_splitter.split(dataset, subject=specific_subject_id):
    # This loop will run only for subject 'S01'.
    # print(f"Processing specific subject: {specific_subject_id}")
    # print(f"  Train set size: {len(train_set_s01.info)}, Validation set size: {len(val_set_s01.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out strategy where the data for *each subject* is independently split into training, validation, *and test* sets. The sum of `val_size` and `test_size` must be less than 1.0. If `split_path` is provided and contains existing split files (e.g., `train_subject_X.csv`, `val_subject_X.csv`, `test_subject_X.csv`), those are used; otherwise, new splits are generated per subject and saved. The `split` method can then yield these train/validation/test triplets for all subjects or for a specific one. Stratification can be applied during the splits if a `stratify` column name is provided.

**Parameters**

- **val_size** (`float`): The proportion of each subject's data to include in their respective validation split. Defaults to `0.2`.
- **test_size** (`float`): The proportion of each subject's data to include in their respective test split. Defaults to `0.2`.
- **stratify** (`str`, optional): If not `None`, data for each subject is split in a stratified fashion, using this string as the column name in the dataset's `info` DataFrame for stratification. Defaults to `None`.
- **shuffle** (`bool`): Whether to shuffle each subject's data before splitting it. Defaults to `False`.
- **random_state** (`Union[int, None]`): Controls the shuffling applied to each subject's data. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-subject split files (`train_subject_X.csv`, `val_subject_X.csv`, `test_subject_X.csv`) are stored or will be saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# a 'subject_id' column, and a 'label' column if stratify='label' is used.

# Example: For each subject, create train/val/test splits (e.g., 60/20/20).
per_subject_tvt_splitter = HoldOutPerSubjectET(
    val_size=0.2,
    test_size=0.2,
    stratify='label', # Optional: stratify by 'label' within each subject's data
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_PER_SUBJECT_TVT_SPLIT_FILES' # Replace with actual path
)

# Iterate through splits for all subjects:
# for train_s, val_s, test_s in per_subject_tvt_splitter.split(dataset):
#     # This loop runs for each subject.
#     # 'train_s', 'val_s', 'test_s' are BaseDataset instances for one subject.
#     # print(f"Processing subject: {train_s.info['subject_id'].iloc[0]}")
#     # print(f"  Train: {len(train_s.info)}, Val: {len(val_s.info)}, Test: {len(test_s.info)}")
#     pass

# Example: Get split for a specific subject (e.g., subject 'S02')
specific_subject_id_et = 'S02' # Replace with an actual subject ID string
for train_s02, val_s02, test_s02 in per_subject_tvt_splitter.split(dataset, subject=specific_subject_id_et):
    # This loop will run only for subject 'S02'.
    # print(f"Processing specific subject: {specific_subject_id_et}")
    # print(f"  Train: {len(train_s02.info)}, Val: {len(val_s02.info)}, Test: {len(test_s02.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a hold-out validation strategy applied individually for each subject, where the split *within each subject's data* is based on a specified grouping column (e.g., 'trial_id', 'session_id'). This ensures that for any given subject, all data points belonging to the same secondary group (e.g., all data from one trial) are entirely within either that subject's training portion or validation portion. If `split_path` is provided and contains existing split files (e.g., `train_subject_X.csv`, `val_subject_X.csv`), those are used; otherwise, new splits are generated for each subject based on their data and the `group_by` criterion, then saved. The `split` method yields these train/validation pairs for each subject (or a specific one).

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame to use for grouping data *within each subject* before splitting (e.g., 'trial_id', 'session_id').
- **val_size** (`float`): The proportion of the unique groups (defined by `group_by`) within each subject's data to include in their respective validation split. Should be between 0.0 and 1.0. Defaults to `0.2`.
- **shuffle** (`bool`): Whether to shuffle the groups (defined by `group_by`) within each subject's data before splitting them. Defaults to `False`.
- **random_state** (`Union[int, None]`): Controls the shuffling of groups within each subject. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-subject split files (e.g., `train_subject_X.csv`, `val_subject_X.csv`) are stored or will be saved. These files will define the train/validation split of groups *within* each subject. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# a 'subject_id' column and the column specified by 'group_by' (e.g., 'trial_id').

# Example: For each subject, split their trials ('group_by') such that
# 20% of trials from that subject go into that subject's validation set.
per_subject_group_splitter = HoldOutPerSubjectCross(
    group_by='trial_id',   # Group data by trial within each subject
    val_size=0.20,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_PER_SUBJECT_GROUPED_SPLIT_FILES' # Replace
)

# Iterate through splits for all subjects:
# 'train_set' and 'val_set' will be BaseDataset instances with their '.info'
# attributes specific to one subject's train/val portion, where the split
# was made by keeping trials together.
# for train_set, val_set, test_set_passthrough in per_subject_group_splitter.split(dataset):
#     # This loop runs for each subject.
#     # current_subject_id = train_set.info['subject_id'].unique()[0]
#     # print(f"Processing subject: {current_subject_id}")
#     # print(f"  Train set trials: {sorted(train_set.info['trial_id'].unique())}")
#     # print(f"  Validation set trials: {sorted(val_set.info['trial_id'].unique())}")
#     pass

# Example: Get split for a specific subject (e.g., subject 'S01')
specific_subject_id = 'S01' # Replace with an actual subject ID string
for train_set_s01, val_set_s01, test_set_s01 in per_subject_group_splitter.split(dataset, subject=specific_subject_id):
    # This loop will run only for subject 'S01'.
    # print(f"Processing specific subject: {specific_subject_id}")
    # print(f"  Train set trials: {sorted(train_set_s01.info['trial_id'].unique())}")
    # print(f"  Validation set trials: {sorted(val_set_s01.info['trial_id'].unique())}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements K-fold cross-validation. This class uses `sklearn.model_selection.KFold` to divide the dataset's `info` DataFrame into `n_splits` folds. For each fold, one part is used as the validation set and the remaining parts are used as the training set. If `split_path` is provided and contains existing split files (e.g., `train_fold_0.csv`, `val_fold_0.csv`), those splits are used; otherwise, new splits are generated for each fold based on the dataset's `info` DataFrame and saved to `split_path`. The `split` method then yields these train/validation pairs for each fold.

**Parameters**

- **n_splits** (`int`): The number of folds. Must be at least 2. Defaults to `5`.
- 

- **shuffle** (`bool`): Whether to shuffle the data before splitting into batches. Note that the samples within each split will not be shuffled. Defaults to `False`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, `random_state` affects the ordering of the indices, which controls the randomness of each fold. Pass an int for reproducible output across multiple function calls. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-fold split files (e.g., `train_fold_X.csv`, `val_fold_X.csv`) are stored or will be saved. If the path does not exist or does not contain appropriately named files for all folds, new splits are generated and saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame.

# Example: Perform 5-fold cross-validation.
k_fold_splitter = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_KFOLD_SPLIT_FILES' # Replace with actual directory path
)

# The 'split' method is a generator, yielding a train/validation pair for each fold.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated for the current fold.
# 'test_set_passthrough' will be None if no test_dataset was passed to .split().
fold_counter = 0
for train_set, val_set, test_set_passthrough in k_fold_splitter.split(dataset):
    fold_counter += 1
    # print(f"Processing Fold {fold_counter}/{k_fold_splitter.n_splits}")
    # print(f"  Train set size: {len(train_set.info)}")
    # print(f"  Validation set size: {len(val_set.info)}")
    # Use train_set and val_set for training and validation for this fold.
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements K-fold cross-validation by splitting based on unique groups identified by the `group_by` column (e.g., 'subject_id', 'session_id'). This ensures that all data points belonging to the same group are assigned to the same fold, either for training or validation, across all K splits. This is particularly useful for preventing data leakage when samples are not independent (e.g., multiple trials from the same subject). The actual K-fold splitting of group IDs is performed using `sklearn.model_selection.KFold`. If `split_path` is provided and contains existing split files for each fold, those are used; otherwise, new splits are generated and saved.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame to use for grouping the data. The K-fold split will be performed on the unique values (groups) found in this column.
- **n_splits** (`int`): The number of folds. Must be at least 2. Defaults to `5`.
- **shuffle** (`bool`): Whether to shuffle the unique groups before splitting them into batches (folds). Defaults to `False`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, `random_state` affects the ordering of the groups, which controls the randomness of each fold. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-fold split files (e.g., `train_fold_X.csv`, `val_fold_X.csv`) are stored or will be saved. These files will define the train/validation split of groups for each fold. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the column specified by 'group_by' (e.g., 'subject_id').

# Example: Perform 5-fold cross-validation, grouping by 'subject_id'.
# This means subjects will be assigned to folds, not individual samples.
k_fold_group_splitter = KFoldCross(
    group_by='subject_id', # Group data by subject for K-fold splitting
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_GROUPED_KFOLD_SPLITS' # Replace with actual directory path
)

# The 'split' method is a generator, yielding a train/validation pair for each fold.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated for the current fold, containing all data for the selected groups.
fold_counter = 0
for train_set, val_set, test_set_passthrough in k_fold_group_splitter.split(dataset):
    fold_counter += 1
    # print(f"Processing Fold {fold_counter}/{k_fold_group_splitter.n_splits}")
    # print(f"  Train set groups (e.g., subjects): {sorted(train_set.info['subject_id'].unique())}")
    # print(f"  Validation set groups (e.g., subjects): {sorted(val_set.info['subject_id'].unique())}")
    # Use train_set and val_set for training and validation for this fold.
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements K-fold cross-validation with a hierarchical grouping structure. The process first iterates through each unique `base_group` (e.g., 'subject_id'). Within each `base_group`, it then identifies unique secondary groups based on the `group_by` column (e.g., 'trial_id' or 'session_id'). For each of these secondary groups, K-fold splitting is applied to its constituent data samples. Finally, the corresponding train and validation sets from all secondary groups (across all base groups) are aggregated to form the K final folds. This ensures that data from each secondary group is distributed across the K folds. If `split_path` is provided and contains existing split files for each fold, those are used; otherwise, new splits are generated and saved.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame used for the secondary level of grouping (e.g., 'trial_id', 'session_id'). K-fold splitting will be applied to the data samples *within* each of these groups.
- **base_group** (`str`): The column name in the dataset's `info` DataFrame used for the primary, coarser level of grouping (e.g., 'subject_id'). The splitting process is iterated for each unique value in this base group. Defaults to `'subject_id'`.
- **n_splits** (`int`): The number of folds for the K-fold split applied within each secondary group. Must be at least 2. Defaults to `5`.
- **shuffle** (`bool`): Whether to shuffle the data samples within each secondary group before splitting them into K folds. Defaults to `False`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, `random_state` affects the ordering of samples within each secondary group, controlling the randomness of each fold. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-fold split files (e.g., `train_fold_X.csv`, `val_fold_X.csv`) are stored or will be saved. These files will contain the aggregated train/validation splits. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the columns specified by 'base_group' (e.g., 'subject_id') and 
# 'group_by' (e.g., 'trial_id').

# Example: For each subject ('base_group'), and for each trial ('group_by') 
# of that subject, apply 5-fold CV to the samples of that trial.
# The final K folds will be aggregations of these splits.
hierarchical_kfold_splitter = KFoldGroupby(
    group_by='trial_id',      # K-fold within each trial
    base_group='subject_id',  # Iterate over subjects
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_HIERARCHICAL_KFOLD_SPLITS' # Replace with actual path
)

# The 'split' method is a generator, yielding a train/validation pair for each of the K aggregated folds.
# 'train_set' and 'val_set' will be new BaseDataset instances with their '.info'
# attributes updated for the current aggregated fold.
fold_counter = 0
for train_set, val_set, test_set_passthrough in hierarchical_kfold_splitter.split(dataset):
    fold_counter += 1
    # print(f"Processing Aggregated Fold {fold_counter}/{hierarchical_kfold_splitter.n_splits}")
    # print(f"  Train set size: {len(train_set.info)}")
    # print(f"  Validation set size: {len(val_set.info)}")
    # Use train_set and val_set for training and validation for this aggregated fold.
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements K-fold cross-validation on a per-subject basis. For each subject identified by 'subject_id' in the dataset's `info` DataFrame, their individual data is split into `n_splits` folds using `sklearn.model_selection.KFold`. This means that for each subject, their data samples are used to create K different train/validation splits specific to that subject. If `split_path` is provided and contains existing split files (e.g., `train_subject_X_fold_Y.csv`), those are used; otherwise, new splits are generated for each subject and each fold, then saved. The `split` method yields these train/validation pairs, iterating through all folds for all subjects (or for a specifically requested subject).

**Parameters**

- **n_splits** (`int`): The number of folds to create for each subject's data. Must be at least 2. Defaults to `5`.
- **shuffle** (`bool`): Whether to shuffle the data for each subject before splitting it into batches (folds). Note that the samples within each split will not be shuffled further by this parameter. Defaults to `False`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, `random_state` affects the ordering of the indices for each subject's data, which controls the randomness of each fold for that subject. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-subject, per-fold split files (e.g., `train_subject_X_fold_Y.csv`, `val_subject_X_fold_Y.csv`) are stored or will be saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# a 'subject_id' column.

# Example: For each subject, perform 5-fold cross-validation on their data.
per_subject_kfold_splitter = KFoldPerSubject(
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_PER_SUBJECT_KFOLD_SPLITS' # Replace with actual path
)

# Iterate through splits for all subjects and all folds:
# 'train_set' and 'val_set' will be BaseDataset instances with their '.info'
# attributes specific to one subject's data for a particular fold.
# for train_set, val_set, test_set_passthrough in per_subject_kfold_splitter.split(dataset):
#     # This loop will run n_splits times for each subject.
#     # current_subject_id = train_set.info['subject_id'].unique()[0] # Example to get current subject
#     # print(f"Processing subject: {current_subject_id}, Fold: ...") # Need a way to get fold_id here if desired
#     # print(f"  Train set size: {len(train_set.info)}, Validation set size: {len(val_set.info)}")
#     pass

# Example: Get all K-fold splits for a specific subject (e.g., subject 'S01')
specific_subject_id = 'S01' # Replace with an actual subject ID string
fold_count_s01 = 0
for train_set_s01, val_set_s01, test_s01 in per_subject_kfold_splitter.split(dataset, subject=specific_subject_id):
    fold_count_s01 += 1
    # This loop will run n_splits times, only for subject 'S01'.
    # print(f"Processing subject: {specific_subject_id}, Fold: {fold_count_s01}")
    # print(f"  Train set size: {len(train_set_s01.info)}, Validation set size: {len(val_set_s01.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements K-fold cross-validation on a per-subject basis, where the splits *within each subject's data* are determined by grouping based on a specified `group_by` column (e.g., 'trial_id', 'session_id'). For each subject, their unique groups (e.g., trials) are divided into `n_splits` folds. This ensures that all data points belonging to a specific group (e.g., one trial) for a particular subject are kept together in either the training or validation set for each fold of that subject's cross-validation. If `split_path` is provided and contains existing split files (e.g., `train_subject_X_fold_Y.csv`), those are used; otherwise, new splits are generated and saved. The `split` method then yields train/validation pairs, iterating through all folds for all subjects (or for a specifically requested subject).

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame used for grouping data *within each subject*. The K-fold split will be performed on these unique intra-subject groups.
- **n_splits** (`int`): The number of folds to create for the groups within each subject's data. Must be at least 2. Defaults to `5`.
- **shuffle** (`bool`): Whether to shuffle the unique groups (defined by `group_by`) within each subject's data before splitting them into folds. Defaults to `False`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, `random_state` affects the ordering of the groups within each subject, controlling the randomness of each fold for that subject. Pass an int for reproducible output. Defaults to `None`.
- **split_path** (`Union[None, str]`): Path to the directory where per-subject, per-fold split files (e.g., `train_subject_X_fold_Y.csv`, `val_subject_X_fold_Y.csv`) are stored or will be saved. These files define the train/validation split of *groups within each subject* for each fold. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# a 'subject_id' column and the column specified by 'group_by' (e.g., 'trial_id').

# Example: For each subject, perform 5-fold cross-validation on their trials.
# This means trials (not individual samples) are assigned to folds for each subject.
per_subject_trial_kfold_splitter = KFoldPerSubjectCross(
    group_by='trial_id', # Group by trial within each subject for K-fold
    n_splits=5,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_PER_SUBJECT_TRIAL_KFOLD_SPLITS' # Replace
)

# Iterate through splits for all subjects and all folds:
# 'train_set' and 'val_set' will be BaseDataset instances with their '.info'
# attributes specific to one subject's data for a particular fold, where trials
# were the units assigned to train/validation for that fold.
# for train_set, val_set, test_set_passthrough in per_subject_trial_kfold_splitter.split(dataset):
#     # This loop runs n_splits times for each subject.
#     # current_subject_id = train_set.info['subject_id'].unique()[0]
#     # print(f"Processing subject: {current_subject_id}, Fold: ... ") # Fold ID isn't directly yielded
#     # print(f"  Train set trials: {sorted(train_set.info['trial_id'].unique())}")
#     # print(f"  Validation set trials: {sorted(val_set.info['trial_id'].unique())}")
#     pass

# Example: Get all K-fold splits for a specific subject's trials (e.g., subject 'S01')
specific_subject_id = 'S01' # Replace with an actual subject ID string
fold_count_s01 = 0
for train_set_s01, val_set_s01, test_s01 in per_subject_trial_kfold_splitter.split(dataset, subject=specific_subject_id):
    fold_count_s01 += 1
    # This loop will run n_splits times, only for subject 'S01'.
    # print(f"Processing subject: {specific_subject_id}, Fold: {fold_count_s01}")
    # print(f"  Train set trials for S01: {sorted(train_set_s01.info['trial_id'].unique())}")
    # print(f"  Validation set trials for S01: {sorted(val_set_s01.info['trial_id'].unique())}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements Leave-One-Group-Out cross-validation. In this strategy, if there are N unique groups (e.g., subjects, sessions, or trials as defined by the `group_by` column), N iterations are performed. In each iteration `i`, the i-th group is held out as the validation set, and all other N-1 groups form the training set. If `split_path` is provided and contains existing split files (e.g., `train_groupX_Y.csv`, `val_groupX_Y.csv`), those are used; otherwise, new splits are generated and saved. The `split` method can then yield these train/validation pairs for all groups or for a specifically requested group to be left out.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame to use for grouping. Each unique value in this column will form a group to be left out in one iteration.
- **split_path** (`Union[None, str]`): Path to the directory where split files (named based on the `group_by` column and the group left out) are stored or will be saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the column specified by 'group_by' (e.g., 'subject_id').

# Example: Perform Leave-One-Subject-Out cross-validation.
leave_one_subject_out_splitter = LeaveOneOut(
    group_by='subject_id', # Each subject will be a validation set once
    split_path='PATH_TO_YOUR_LOO_SPLIT_FILES' # Replace with actual directory path
)

# Iterate through splits, leaving one subject out each time:
# 'train_set' contains all other subjects, 'val_set' contains the left-out subject.
# 'test_set_passthrough' will be None if no test_dataset was passed to .split().
# for train_set, val_set, test_set_passthrough in leave_one_subject_out_splitter.split(dataset):
#     # This loop runs for each unique subject_id in the dataset.
#     # left_out_subject = val_set.info['subject_id'].unique()[0]
#     # print(f"Leaving out subject: {left_out_subject}")
#     # print(f"  Train set size: {len(train_set.info)}, Validation set size: {len(val_set.info)}")
#     pass

# Example: Get split where a specific subject (e.g., 'S03') is left out for validation
specific_group_to_leave_out = 'S03' # Replace with an actual group ID (e.g., subject ID)
for train_set_loo, val_set_loo, test_set_loo in leave_one_subject_out_splitter.split(dataset, group=specific_group_to_leave_out):
    # This loop will run only once for the specified group.
    # print(f"Specific group '{specific_group_to_leave_out}' left out for validation.")
    # print(f"  Train set size: {len(train_set_loo.info)}, Validation set size: {len(val_set_loo.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a Leave-One-Out style cross-validation that generates training, validation, and test sets for each iteration. In this strategy, for each unique group (defined by `group_by`, e.g., 'subject_id') designated as the `test_group` in an iteration:

1. The `test_group` itself forms the test set.
2. The group immediately preceding the `test_group` (cyclically from the sorted list of all unique groups) is chosen as the validation set (`val_group`).
3. All other remaining groups form the training set. This process is repeated, with each group taking a turn as the `test_group`. If `split_path` is provided and contains existing split files (e.g., `train_groupX_Y.csv`, `val_groupX_Y.csv`, `test_groupX_Y.csv` where Y is the `test_group`), those are used; otherwise, new splits are generated and saved.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame to use for grouping. Each unique value in this column will define a group.
- **split_path** (`Union[None, str]`): Path to the directory where split files (named based on the `group_by` column and the group designated as the `test_group` for that iteration) are stored or will be saved. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the column specified by 'group_by' (e.g., 'session_id').

# Example: Perform Leave-One-Session-Out, also selecting a validation session.
loo_et_splitter = LeaveOneOutEAndT(
    group_by='session_id', # Each session will serve as the test set once
    split_path='PATH_TO_YOUR_LOO_ET_SPLIT_FILES' # Replace with actual directory path
)

# Iterate through splits:
# In each iteration:
# - 'test_s' contains the session currently left out as the test set.
# - 'val_s' contains the session preceding the test_s (cyclically) as the validation set.
# - 'train_s' contains all other sessions.
# for train_s, val_s, test_s in loo_et_splitter.split(dataset):
#     # This loop runs for each unique session_id in the dataset.
#     # current_test_session = test_s.info['session_id'].unique()[0]
#     # current_val_session = val_s.info['session_id'].unique()[0]
#     # print(f"Test session: {current_test_session}, Val session: {current_val_session}")
#     # print(f"  Train: {len(train_s.info)}, Val: {len(val_s.info)}, Test: {len(test_s.info)}")
#     pass

# Example: Get a specific split where 'Session3' is the test group.
# The validation group will be the one preceding 'Session3'.
specific_test_group = 'Session3' # Replace with an actual group ID
for train_specific, val_specific, test_specific in loo_et_splitter.split(dataset, group=specific_test_group):
    # This loop will run only once for the specified test group.
    # print(f"Test group: {specific_test_group}")
    # print(f"  Train: {len(train_specific.info)}, Val: {len(val_specific.info)}, Test: {len(test_specific.info)}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

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

Implements a "Leave-One-Session-Out" (LOSO)-like rotating cross-validation strategy. The process involves two main stages:

1. **Coarse K-Fold Split**: Unique groups (defined by the `group_by` column, e.g., 'session_id' or 'subject_id') are first divided into `n_coarse_splits` folds. In each iteration of this coarse split, one fold becomes the "rest set" and the others form the initial part of the training set.

2. Rotating Validation/Test Sets

   : For each "rest set" obtained from the coarse split, a rotation mechanism is applied. In each step of the rotation:

   - A specified number of groups (`num_test_groups_in_rotation`) from the end of the (cyclically shifted) "rest set" are designated as the test set for the current generated fold.
   - A specified number of groups (`num_val_groups_in_rotation`) immediately preceding the test set groups are designated as the validation set.
   - The remaining groups from the "rest set", along with the initial training groups from the coarse split, form the final training set for this generated fold. This process generates multiple train/validation/test splits. If `split_path` is provided and contains existing appropriately named split files (e.g., `train_fold_X.csv`), those are used; otherwise, new splits are generated and saved.

**Parameters**

- **group_by** (`str`): The column name in the dataset's `info` DataFrame used to define the unique groups that will be split and rotated (e.g., 'session_id', 'subject_id').
- **split_path** (`Union[None, str]`): Path to the directory where split definition files (e.g., `train_fold_X.csv`, `val_fold_X.csv`, `test_fold_X.csv`) are stored or will be saved. Defaults to `None`.
- **n_coarse_splits** (`int`): The number of folds for the initial coarse K-Fold splitting of the unique groups. Defaults to `4`.
- **num_val_groups_in_rotation** (`int`): The number of groups from the "rest set" of the coarse split to be used for the validation set in each rotation. Defaults to `2`.
- **num_test_groups_in_rotation** (`int`): The number of groups from the "rest set" of the coarse split to be used for the test set in each rotation. Must be greater than 0. Defaults to `1`.
- **shuffle** (`bool`): Whether to shuffle the unique groups before the initial coarse K-Fold splitting. Defaults to `True`.
- **random_state** (`Union[None, int]`): When `shuffle` is `True`, this seed is used for the coarse K-Fold shuffling, ensuring reproducibility. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'dataset' is an existing instance of BaseDataset.
# 'dataset.info' is expected to be a pandas DataFrame and must contain
# the column specified by 'group_by' (e.g., 'session_id').

# Example: Implement rotating cross-split, grouping by 'session_id'.
# Coarsely split sessions into 4 folds. From the "left-out" sessions of each coarse fold,
# rotate them such that 1 session is for test, and 2 sessions before it are for validation.
rotating_splitter = LosoRotatingCrossSplit(
    group_by='session_id',
    n_coarse_splits=4,
    num_val_groups_in_rotation=2,
    num_test_groups_in_rotation=1,
    shuffle=True,
    random_state=42,
    split_path='PATH_TO_YOUR_LOSO_ROTATING_SPLITS' # Replace with actual path
)

# The 'split' method is a generator, yielding a train/validation/test triplet for each generated fold.
# The number of yielded folds will be n_coarse_splits * (number of groups in each "rest set").
fold_counter = 0
for train_set, val_set, test_set in rotating_splitter.split(dataset):
    fold_counter += 1
    # print(f"Processing Generated Fold {fold_counter}")
    # print(f"  Train set groups (e.g., sessions): {sorted(train_set.info['session_id'].unique())}")
    # print(f"  Validation set groups: {sorted(val_set.info['session_id'].unique())}")
    # print(f"  Test set groups: {sorted(test_set.info['session_id'].unique())}")
    # Ensure no overlap between train, val, test groups for this fold.
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)

## NoSplit

~~~python
class NoSplit(BaseSplit):
    def __init__(
        self, 
        split_path: str = None,
        **kwargs
    ) -> None:
~~~

This class represents a splitting strategy where no actual splitting of the primary `dataset` occurs. Instead, it directly yields the provided `dataset`, `val_dataset`, and `test_dataset` as a single "split". This is useful in scenarios where pre-split datasets are already available and need to be used within a framework expecting a split-generating mechanism, or when evaluating on the full training set (though typically a validation set would still be distinct). The `split_path` parameter is inherited but not actively used by this class as no split files are generated or read.

**Parameters**

- **split_path** (`str`, optional): Path to split dataset information. This parameter is inherited from `BaseSplit` but is not used by the `NoSplit` class as no actual splitting or loading/saving of split files occurs. Defaults to `None`.
- ***\*kwargs**: Additional keyword arguments passed to the parent `BaseSplit` class.

**Usage Example**

~~~python
# Assume 'train_dataset_predefined', 'validation_dataset_predefined', 
# and 'test_dataset_predefined' are existing instances of BaseDataset.

# 1. Instantiate the NoSplit strategy.
#    'split_path' is not used by NoSplit but is part of the BaseSplit signature.
no_split_strategy = NoSplit(split_path=None)

# 2. "Split" the datasets.
#    This will yield the original datasets back as a single tuple.
for train_set, val_set, test_set in no_split_strategy.split(
    dataset=train_dataset_predefined, 
    val_dataset=validation_dataset_predefined, 
    test_dataset=test_dataset_predefined
):
    # 'train_set' will be the same as 'train_dataset_predefined'.
    # 'val_set' will be the same as 'validation_dataset_predefined'.
    # 'test_set' will be the same as 'test_dataset_predefined'.
    
    # Example: print basic info about the yielded datasets
    # if train_set:
    #     print(f"Yielded Training set info length: {len(train_set.info) if hasattr(train_set, 'info') else 'N/A'}")
    # if val_set:
    #     print(f"Yielded Validation set info length: {len(val_set.info) if hasattr(val_set, 'info') else 'N/A'}")
    # if test_set:
    #     print(f"Yielded Test set info length: {len(test_set.info) if hasattr(test_set, 'info') else 'N/A'}")
    pass
~~~

[`Back to Top`](#tyeedatasetsplit)