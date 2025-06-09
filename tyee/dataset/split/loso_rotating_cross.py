import logging
import os
import re
from copy import copy
from typing import Dict, Tuple, Union, Generator, List, Optional

import numpy as np
import pandas as pd
from sklearn import model_selection

from dataset.base_dataset import BaseDataset
from .base_split import BaseSplit

log = logging.getLogger('split')


class LosoRotatingCrossSplit(BaseSplit):
    def __init__(
        self,
        group_by: str,
        split_path: Union[None, str] = None,
        n_coarse_splits: int = 4,  # Corresponds to KFold(n_splits=4) in original
        num_val_groups_in_rotation: int = 2, # Corresponds to rest[-3:-1] (2 groups)
        num_test_groups_in_rotation: int = 1, # Corresponds to rest[-1:] (1 group)
        shuffle: bool = True, # Shuffle for the coarse KFold
        random_state: Union[None, int] = None,
        **kwargs
    ) -> None:
        """
        Initialize the LosoRotatingCrossSplit class.
        This method implements a Leave-One-Session-Out (LOSO) like split based on
        an initial coarse KFold split, followed by rotating subsets for validation and testing.

        Args:
            group_by (str): The column name in the info DataFrame to group unique entities by
                            (e.g., 'subject_id', 'session_id').
            split_path (Union[None, str]): Path to save/load split definition files.
            n_coarse_splits (int): Number of folds for the initial coarse splitting of groups.
                                   Similar to the KFold(n_splits=4) in the original example.
            num_val_groups_in_rotation (int): Number of groups from the 'rest' set to use for validation
                                              in each rotation. (e.g., 2 from original: rest[-3:-1]).
            num_test_groups_in_rotation (int): Number of groups from the 'rest' set to use for testing
                                               in each rotation. (e.g., 1 from original: rest[-1:]).
            shuffle (bool): Whether to shuffle the groups before the coarse KFold splitting.
            random_state (Union[None, int]): Random seed for reproducibility of the coarse shuffle.
        """
        super().__init__(split_path=split_path, **kwargs)
        self.group_by = group_by
        self.n_coarse_splits = n_coarse_splits
        self.num_val_groups_in_rotation = num_val_groups_in_rotation
        self.num_test_groups_in_rotation = num_test_groups_in_rotation
        self.shuffle = shuffle
        self.random_state = random_state

        if self.n_coarse_splits < 2 and self.n_coarse_splits != -1: # -1 might mean leave one out if all groups are few
             # Or handle cases where n_coarse_splits might be 1 if only one rotation is desired (less common)
            log.warning(
                f'Number of coarse splits is {self.n_coarse_splits}. '
                f'Typical K-Fold requires at least 2 splits.'
            )
        if self.num_val_groups_in_rotation < 0 or self.num_test_groups_in_rotation <= 0:
            raise ValueError("Number of validation/test groups in rotation must be positive (test > 0).")


        self.coarse_k_fold = model_selection.KFold(
            n_splits=self.n_coarse_splits,
            shuffle=self.shuffle,
            random_state=self.random_state
        )
        self._fold_generation_complete = False


    def split_info_constructor(self, info: pd.DataFrame) -> None:
        """
        Create train, validation, and test split files for each generated fold.
        The splits are defined by group IDs based on the `group_by` column.

        Args:
            info (pd.DataFrame): DataFrame containing dataset information, including the `group_by` column.
        """
        if not self.split_path:
            raise ValueError("split_path must be set to save split information.")
        if not os.path.exists(self.split_path):
            os.makedirs(self.split_path)

        unique_group_ids = np.array(sorted(list(set(info[self.group_by]))))

        if len(unique_group_ids) < self.n_coarse_splits:
            log.warning(
                f"Number of unique groups ({len(unique_group_ids)}) is less than n_coarse_splits ({self.n_coarse_splits}). "
                f"Adjusting n_coarse_splits to {len(unique_group_ids)} (LeaveOneGroupOut for coarse split)."
            )
            # Fallback to LOO for coarse split if not enough groups
            # Or could raise error, depending on desired behavior.
            # For this implementation, let's adjust n_splits for KFold if it's too high.
            # However, KFold itself will raise an error if n_splits > n_samples.
            # So, it's better to check upfront or ensure n_coarse_splits is appropriate.
            # For simplicity, the KFold will handle this error, or we ensure n_coarse_splits <= len(unique_group_ids)
            if len(unique_group_ids) < 2 : # KFold needs at least 2 samples for n_splits >= 2
                raise ValueError(f"Not enough unique groups ({len(unique_group_ids)}) to perform a split.")
            current_coarse_k_fold = model_selection.KFold(
                n_splits=min(self.n_coarse_splits, len(unique_group_ids)), # Adjust n_splits
                shuffle=self.shuffle,
                random_state=self.random_state
            )
        else:
            current_coarse_k_fold = self.coarse_k_fold


        fold_counter = 0
        required_in_rest = self.num_val_groups_in_rotation + self.num_test_groups_in_rotation

        for coarse_train_indices, coarse_rest_indices in current_coarse_k_fold.split(unique_group_ids):
            coarse_train_set_group_ids = unique_group_ids[coarse_train_indices]
            rest_set_group_ids = unique_group_ids[coarse_rest_indices]

            if len(rest_set_group_ids) < required_in_rest:
                log.warning(
                    f"Coarse fold resulted in a 'rest' set of size {len(rest_set_group_ids)}, "
                    f"which is smaller than the required {required_in_rest} groups for validation and testing. "
                    f"Skipping rotations for this coarse fold."
                )
                continue

            current_rest_for_rotation = np.array(list(rest_set_group_ids)) # Mutable copy for rolling

            for i in range(len(current_rest_for_rotation)): # Rotate for each item in the rest set
                # Determine test, validation, and remaining train from current_rest_for_rotation
                # Test: last `num_test_groups_in_rotation` elements
                test_ids_from_rotation = current_rest_for_rotation[-self.num_test_groups_in_rotation:]

                # Validation: elements before test
                val_ids_from_rotation = current_rest_for_rotation[
                    -(self.num_test_groups_in_rotation + self.num_val_groups_in_rotation) :
                    -self.num_test_groups_in_rotation
                ]
                # Ensure val_ids_from_rotation is not empty if num_val_groups_in_rotation is 0
                if self.num_val_groups_in_rotation == 0:
                    val_ids_from_rotation = np.array([])


                # Train from rotation: elements before validation
                train_ids_from_rotation = current_rest_for_rotation[
                    :-(self.num_test_groups_in_rotation + self.num_val_groups_in_rotation)
                ]

                # Combine train parts
                final_train_group_ids = np.concatenate(
                    (coarse_train_set_group_ids, train_ids_from_rotation)
                ).tolist()
                final_val_group_ids = val_ids_from_rotation.tolist()
                final_test_group_ids = test_ids_from_rotation.tolist()

                # --- Assertions for consistency (optional but good for debugging) ---
                all_split_ids = set(final_train_group_ids) | set(final_val_group_ids) | set(final_test_group_ids)
                original_ids_in_this_config = set(coarse_train_set_group_ids) | set(rest_set_group_ids)

                if not (set(final_train_group_ids).isdisjoint(set(final_val_group_ids)) and \
                        set(final_train_group_ids).isdisjoint(set(final_test_group_ids)) and \
                        set(final_val_group_ids).isdisjoint(set(final_test_group_ids))):
                    log.error(f"Overlap detected in fold {fold_counter}!")
                    # Handle error or debug

                if not all_split_ids == original_ids_in_this_config:
                     log.error(f"Mismatch in group IDs for fold {fold_counter}!")
                     # Handle error or debug
                # --- End Assertions ---

                train_info_df = info[info[self.group_by].isin(final_train_group_ids)].copy()
                val_info_df = info[info[self.group_by].isin(final_val_group_ids)].copy()
                test_info_df = info[info[self.group_by].isin(final_test_group_ids)].copy()

                # Check for empty splits which can happen if groups are missing from `info`
                if train_info_df.empty or (val_info_df.empty and self.num_val_groups_in_rotation > 0) or test_info_df.empty:
                    log.warning(f"Fold {fold_counter} resulted in one or more empty dataframes. "
                                f"Train: {len(train_info_df)}, Val: {len(val_info_df)}, Test: {len(test_info_df)}. "
                                f"Group IDs - Train: {len(final_train_group_ids)}, Val: {len(final_val_group_ids)}, Test: {len(final_test_group_ids)}. "
                                "This might be due to filtering or missing group IDs in the main 'info' DataFrame.")
                    # Optionally, skip saving this fold or handle as an error
                    current_rest_for_rotation = np.roll(current_rest_for_rotation, 1) # Critical: roll for next iteration
                    continue


                train_info_df.to_csv(os.path.join(self.split_path, f'train_fold_{fold_counter}.csv'), index=False)
                val_info_df.to_csv(os.path.join(self.split_path, f'val_fold_{fold_counter}.csv'), index=False)
                test_info_df.to_csv(os.path.join(self.split_path, f'test_fold_{fold_counter}.csv'), index=False)

                fold_counter += 1
                current_rest_for_rotation = np.roll(current_rest_for_rotation, 1) # Roll for next iteration

        if fold_counter == 0:
            log.warning("No folds were generated. Check group sizes and split parameters.")
        self._fold_generation_complete = True
        log.info(f"Generated {fold_counter} folds in total.")


    @property
    def fold_ids(self) -> List[int]:
        """
        Retrieve the list of fold IDs based on existing split files (e.g., train_fold_X.csv).

        Returns:
            List[int]: Sorted list of unique fold IDs found in the split_path.
        """
        if not self.split_path or not os.path.exists(self.split_path):
            log.warning(f"Split path '{self.split_path}' does not exist or not set. Cannot determine fold IDs.")
            return []

        indice_files = os.listdir(self.split_path)
        found_fold_ids = set()

        for f_name in indice_files:
            # Look for train, val, or test files to extract fold ID
            match = re.search(r'(?:train|val|test)_fold_(\d+)\.csv', f_name)
            if match:
                found_fold_ids.add(int(match.group(1)))

        if not found_fold_ids and not self._fold_generation_complete :
            log.info("No fold files found. You might need to run split_info_constructor first or check path.")
        elif not found_fold_ids and self._fold_generation_complete:
             log.warning("Fold generation was marked complete, but no fold files were found. This might indicate an issue during generation.")


        return sorted(list(found_fold_ids))

    def split(
        self,
        dataset: BaseDataset,
        # val_dataset and test_dataset params are not used here, as splits are derived from the main dataset
        # For consistency with KFoldCross, they are kept but ignored.
        val_dataset_ignored: Optional[BaseDataset] = None,
        test_dataset_ignored: Optional[BaseDataset] = None,
    ) -> Generator[Tuple[BaseDataset, BaseDataset, BaseDataset], None, None]:
        """
        Generate train, validation, and test datasets for each fold.
        If split files don't exist, it creates them. Otherwise, it loads them.

        Args:
            dataset (BaseDataset): The dataset to split. Its `info` attribute will be used.

        Yields:
            Tuple[BaseDataset, BaseDataset, BaseDataset]: Train, validation, and test datasets
                                                          for each fold.
        """
        if not self.check_split_path(): # check_split_path might need to be more robust
            log.info(f'📊 | Creating new split files in {self.split_path}.')
            if self.split_path:
                 log.info(
                    f'😊 | For future runs with the same splits, ensure \033[92msplit_path\033[0m '
                    f'(\033[92m{self.split_path}\033[0m) is correctly set and populated.'
                )
            else:
                log.error("Split path is not defined. Cannot create or load splits.")
                return # or raise error

            # Ensure dataset.info is available
            if dataset.info is None or dataset.info.empty:
                log.error("Dataset 'info' attribute is missing or empty. Cannot generate splits.")
                return # or raise error

            self.split_info_constructor(dataset.info)
        else:
            log.info(f'📊 | Detected existing split files in {self.split_path}. Loading them.')
            log.info(
                '💡 | If the underlying dataset metadata (dataset.info) has changed significantly, '
                'consider deleting the old split files and re-generating the splits.'
            )
            self._fold_generation_complete = True # Assume complete if files exist

        current_fold_ids = self.fold_ids
        if not current_fold_ids:
            log.error(f"No fold IDs could be determined from {self.split_path}. Cannot proceed with splitting.")
            return

        for fold_id in current_fold_ids:
            try:
                train_info_path = os.path.join(self.split_path, f'train_fold_{fold_id}.csv')
                val_info_path = os.path.join(self.split_path, f'val_fold_{fold_id}.csv')
                test_info_path = os.path.join(self.split_path, f'test_fold_{fold_id}.csv')

                train_info = pd.read_csv(train_info_path, low_memory=False)
                val_info = pd.read_csv(val_info_path, low_memory=False)
                test_info = pd.read_csv(test_info_path, low_memory=False)
            except FileNotFoundError as e:
                log.error(f"Error loading files for fold {fold_id}: {e}. Ensure all (train, val, test) files exist.")
                continue # Skip this fold or raise error

            train_fold_dataset = copy(dataset)
            train_fold_dataset.info = train_info

            val_fold_dataset = copy(dataset)
            val_fold_dataset.info = val_info

            test_fold_dataset = copy(dataset)
            test_fold_dataset.info = test_info

            yield train_fold_dataset, val_fold_dataset, test_fold_dataset

    @property
    def repr_body(self) -> Dict:
        """
        Representation body for the class.
        """
        body = {
            'group_by': self.group_by,
            'split_path': self.split_path,
            'n_coarse_splits': self.n_coarse_splits,
            'num_val_groups_in_rotation': self.num_val_groups_in_rotation,
            'num_test_groups_in_rotation': self.num_test_groups_in_rotation,
            'shuffle': self.shuffle,
            'random_state': self.random_state,
        }
        # Add parent's repr_body if there are relevant attributes there
        # parent_repr = super().repr_body if hasattr(super(), 'repr_body') else {}
        # return {**parent_repr, **body}
        return body
