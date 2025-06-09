from dataset import BaseDataset
from .base_split import BaseSplit
from typing import Generator, Tuple


class NoSplit(BaseSplit):
    def __init__(
        self, 
        split_path: str = None,
        **kwargs
    ) -> None:
        """
        Initialize the NoSplit class.

        Args:
            split_path (str): Path to the split dataset (not used in this class).
        """
        super().__init__(split_path)

    def split(
        self,
        dataset: BaseDataset,
        val_dataset: BaseDataset = None,
        test_dataset: BaseDataset = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Directly return the input datasets without any splitting.

        Args:
            dataset (BaseDataset): The training dataset.
            val_dataset (BaseDataset): The validation dataset.
            test_dataset (BaseDataset): The test dataset.

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: The input datasets as-is.
        """
        yield dataset, val_dataset, test_dataset