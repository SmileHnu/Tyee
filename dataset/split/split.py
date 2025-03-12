#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : split.py
@Time    : 2025/03/05 19:06:04
@Desc    : 
"""


from typing import Dict, List, Tuple, Union
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import Subset
from dataset import BaseDataset
import pandas as pd

class DatasetSplitter:
    """
    数据集划分工具类，用于将数据集划分为训练集、验证集和测试集。

    参数:
        train_dataset (BaseDataset): 训练数据集。
        dev_dataset (BaseDataset, 可选): 验证数据集，默认为 None。
        test_dataset (BaseDataset, 可选): 测试数据集，默认为 None。
    """
    def __init__(self, 
                 train_dataset: BaseDataset, 
                 dev_dataset: BaseDataset = None, 
                 test_dataset: BaseDataset = None):
        self.train_dataset = train_dataset
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset

    def split(self, method: str = 'none', split_by: str = 'record_id', seed: int = None, **kwargs) -> Union[Tuple[BaseDataset, BaseDataset, BaseDataset], List[Tuple[BaseDataset, BaseDataset, BaseDataset]]]:
        """
        划分数据集的方法。

        参数:
            method (str): 划分数据集的方法。可选值为 'none'、'kfold'、'hold_out'。
            split_by (str): 划分数据集的粒度。可选值为 'record_id'、'clip_id'、'subject_id'、'session_id'、'trial_id'。
            seed (int, 可选): 随机种子，用于数据划分的可重复性。
            **kwargs: 其他参数。

        返回:
            Union[Tuple[BaseDataset, BaseDataset, BaseDataset], List[Tuple[BaseDataset, BaseDataset, BaseDataset]]]: 划分后的数据集。
        """
        if method == 'none':
            # 不需要划分数据集
            return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

        elif method == 'kfold':
            # 按k折叉验证的方式划分数据集
            n_splits = kwargs.get('n_splits', 5)
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
            unique_ids = self.train_dataset.info[split_by].unique()
            splits = [(self._create_subset(unique_ids[train_idx], split_by), self._create_subset(unique_ids[val_idx], split_by), self.test_dataset)
                      for train_idx, val_idx in kf.split(unique_ids)]
            return splits

        elif method == 'hold_out':
            # 按留出法划分数据集
            unique_ids = self.train_dataset.info[split_by].unique()
            test_size = kwargs.get('test_size', 0.2)
            dev_size = kwargs.get('dev_size', 0.2)

            if self.dev_dataset is None and self.test_dataset is None:
                train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
                train_ids, dev_ids = train_test_split(train_ids, test_size=dev_size, random_state=seed)
                test_subset = self._create_subset(test_ids, split_by)
            elif self.dev_dataset is None:
                train_ids, dev_ids = train_test_split(unique_ids, test_size=dev_size, random_state=seed)
                print(f'train_ids:{train_ids}')
                print(f'dev_ids:{dev_ids}')
                test_subset = self.test_dataset
            elif self.test_dataset is None:
                train_ids, test_ids = train_test_split(unique_ids, test_size=test_size, random_state=seed)
                test_subset = self._create_subset(test_ids, split_by)
            else:
                return [(self.train_dataset, self.dev_dataset, self.test_dataset)]

            train_subset = self._create_subset(train_ids, split_by)
            val_subset = self._create_subset(dev_ids, split_by)
            return [(train_subset, val_subset, test_subset)]
        # elif method == 'tuev':
        #     train_ids = ['record_0', 'record_1', 'record_2', 'record_3', 'record_4', 'record_5', 
        #                  'record_6', 'record_7', 'record_9', 'record_10', 'record_11', 'record_13', 
        #                  'record_14', 'record_15', 'record_16', 'record_17', 'record_18', 'record_19', 
        #                  'record_21', 'record_22', 'record_23', 'record_24', 'record_25', 'record_30', 
        #                  'record_32', 'record_33', 'record_34', 'record_35', 'record_36', 'record_37', 
        #                  'record_38', 'record_39', 'record_40', 'record_41', 'record_42', 'record_43', 
        #                  'record_44', 'record_45', 'record_46', 'record_47', 'record_49', 'record_50', 
        #                  'record_51', 'record_52', 'record_53', 'record_54', 'record_55', 'record_57', 
        #                  'record_58', 'record_59', 'record_61', 'record_62', 'record_63', 'record_64', 
        #                  'record_65', 'record_66', 'record_68', 'record_69', 'record_70', 'record_71', 
        #                  'record_72', 'record_73', 'record_74', 'record_76', 'record_77', 'record_78', 
        #                  'record_79', 'record_80', 'record_81', 'record_82', 'record_84', 'record_85', 
        #                  'record_86', 'record_87', 'record_88', 'record_89', 'record_90', 'record_93', 
        #                  'record_95', 'record_97', 'record_98', 'record_99', 'record_100', 'record_101',
        #                  'record_102', 'record_103', 'record_104', 'record_105', 'record_106', 'record_107', 
        #                  'record_108', 'record_109', 'record_110', 'record_111', 'record_112', 'record_113', 
        #                  'record_114', 'record_115', 'record_116', 'record_117', 'record_118', 'record_119', 
        #                  'record_121', 'record_123', 'record_124', 'record_126', 'record_127', 'record_128', 
        #                  'record_129', 'record_130', 'record_132', 'record_133', 'record_134', 'record_135', 
        #                  'record_136', 'record_137', 'record_138', 'record_139', 'record_141', 'record_142', 
        #                  'record_143', 'record_145', 'record_147', 'record_148', 'record_149', 'record_150', 
        #                  'record_151', 'record_152', 'record_153', 'record_155', 'record_156', 'record_159', 
        #                  'record_161', 'record_162', 'record_163', 'record_164', 'record_166', 'record_167', 
        #                  'record_168', 'record_169', 'record_171', 'record_172', 'record_173', 'record_174', 
        #                  'record_175', 'record_176', 'record_178', 'record_179', 'record_180', 'record_181', 
        #                  'record_182', 'record_183', 'record_184', 'record_185', 'record_186', 'record_187', 
        #                  'record_188', 'record_189', 'record_190', 'record_191', 'record_192', 'record_193', 
        #                  'record_194', 'record_195', 'record_196', 'record_197', 'record_198', 'record_199', 
        #                  'record_200', 'record_201', 'record_202', 'record_203', 'record_206', 'record_207', 
        #                  'record_208', 'record_209', 'record_210', 'record_211', 'record_212', 'record_213', 
        #                  'record_214', 'record_215', 'record_216', 'record_217', 'record_219', 'record_220', 
        #                  'record_221', 'record_222', 'record_223', 'record_224', 'record_225', 'record_226', 
        #                  'record_228', 'record_229', 'record_230', 'record_232', 'record_237', 'record_238', 
        #                  'record_246', 'record_247', 'record_248', 'record_249', 'record_250', 'record_251', 
        #                  'record_254', 'record_255', 'record_256', 'record_257', 'record_258', 'record_259', 
        #                  'record_260', 'record_261', 'record_262', 'record_263', 'record_264', 'record_265', 
        #                  'record_266', 'record_267', 'record_268', 'record_269', 'record_270', 'record_271', 
        #                  'record_272', 'record_273', 'record_278', 'record_279', 'record_283', 'record_284', 
        #                  'record_285', 'record_286', 'record_287', 'record_288', 'record_289', 'record_290', 
        #                  'record_291', 'record_292', 'record_293', 'record_294', 'record_295', 'record_296', 
        #                  'record_298', 'record_299', 'record_300', 'record_301', 'record_302', 'record_304', 
        #                  'record_306', 'record_307', 'record_308', 'record_309', 'record_310', 'record_312', 
        #                  'record_314', 'record_315', 'record_316', 'record_317', 'record_319', 'record_321', 
        #                  'record_322', 'record_323', 'record_324', 'record_325', 'record_327', 'record_328', 
        #                  'record_329', 'record_330', 'record_331', 'record_332', 'record_333', 'record_334', 
        #                  'record_335', 'record_336', 'record_338', 'record_339', 'record_341', 'record_342', 
        #                  'record_344', 'record_347', 'record_348', 'record_349', 'record_350', 'record_351', 
        #                  'record_354', 'record_355', 'record_357', 'record_358']
        #     dev_ids = ['record_8', 'record_12', 'record_20', 'record_26', 'record_27', 'record_28', 
        #                'record_29', 'record_31', 'record_48', 'record_56', 'record_60', 'record_67', 
        #                'record_75', 'record_83', 'record_91', 'record_92', 'record_94', 'record_96', 
        #                'record_120', 'record_122', 'record_125', 'record_131', 'record_140', 'record_144', 
        #                'record_146', 'record_154', 'record_157', 'record_158', 'record_160', 'record_165', 
        #                'record_170', 'record_177', 'record_204', 'record_205', 'record_218', 'record_227', 
        #                'record_231', 'record_233', 'record_234', 'record_235', 'record_236', 'record_239', 
        #                'record_240', 'record_241', 'record_242', 'record_243', 'record_244', 'record_245', 
        #                'record_252', 'record_253', 'record_274', 'record_275', 'record_276', 'record_277', 
        #                'record_280', 'record_281', 'record_282', 'record_297', 'record_303', 'record_305', 
        #                'record_311', 'record_313', 'record_318', 'record_320', 'record_326', 'record_337', 
        #                'record_340', 'record_343', 'record_345', 'record_346', 'record_352', 'record_353', 
        #                'record_356']
        #     train_subset = self._create_subset(train_ids, split_by)
        #     val_subset = self._create_subset(dev_ids, split_by)
        #     test_subset = self.test_dataset
        #     return [(train_subset, val_subset, test_subset)]
        else:
            raise ValueError(f"Unsupported split method: {method}")


    def _create_subset(self, ids: List[int], split_by: str) -> Subset:
        """
        根据ids创建BaseDataset子集。

        参数:
            ids (List[int]): 样本的ID列表。
            split_by (str): 划分数据集的粒度。

        返回:
            Subset: 子集。
        """
        indices = self.train_dataset.info[self.train_dataset.info[split_by].isin(ids)].index.tolist()
        return Subset(self.train_dataset, indices)