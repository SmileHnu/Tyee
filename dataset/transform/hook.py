#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : shulingyu
@License : (C) Copyright 2025, Hunan University
@Contact : shulingyu@hnu.edu.cn
@Software: Visual Studio Code
@File    : hook.py
@Time    : 2025/03/17 14:21:24
@Desc    : 
"""

from dataset.transform import BaseTransform
from typing import Dict, Any
import numpy as np
import scipy

class EA(BaseTransform):
    def __init__(self,new_R = None):
        """
        欧氏空间对齐
        """
        super().__init__()
        self.new_R = new_R
        
    def transform(self, result: Dict[str, Any]) -> Dict[str, Any]:
        x = result['signals']
        xt = np.transpose(x,axes=(0,2,1))
        # print('xt shape:',xt.shape)
        E = np.matmul(x,xt)
        # print(E.shape)
        R = np.mean(E, axis=0)
        # print('R shape:',R.shape)

        R_mat = scipy.linalg.fractional_matrix_power(R,-0.5)
        new_x = np.einsum('n c s,r c -> n r s',x,R_mat)
        if self.new_R is None:
            return new_x

        new_x = np.einsum('n c s,r c -> n r s',new_x,scipy.linalg.fractional_matrix_power(self.new_R,0.5))
        result['signals'] = new_x
        return result

