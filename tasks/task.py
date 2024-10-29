#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : task.py
@Time    : 2024/09/25 16:41:26
@Desc    : 
"""


class PRLTask(object):
    def __init__(self) -> None:
        self.dataset = None
        self.model = None
        self.optimizer = None
        pass

    def load_data(self, ) -> None:
        pass

    def train_step(self, ) -> None:
        pass

    def valid_step(self, ) -> None:
        pass