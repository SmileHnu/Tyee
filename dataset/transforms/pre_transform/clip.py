#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2024, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : clip.py
@Time    : 2024/11/14 16:05:16
@Desc    : 
"""


class Clip(object):
    def __init__(
            self,
            start: int = 0,
            end: int = -1,
            clip_len: int = -1
        ) -> None:
        self._start = start
        self._end = end
        self._clip_len = clip_len
        pass

    def __call__(self, sample: dict) -> list[dict]:
        x = sample["x"]
        y = sample["y"]
        info = sample["info"]
        fs = info["fs"]

        if self._start > 0 and self._end > self._start:
            s, e = int(self._start * fs), int(self._end * fs)
            x = x[s:e, :]
            return [{
                "id": sample["id"],
                "x": x,
                "y": y,
                "info": info
            }]
        
        if self._clip_len > 0:
            rs = []
            for idx, s in enumerate(range(0, x.shape[0], self._clip_len)):
                rs.append({
                    "id": f"{sample['id']}_{idx}",
                    "x": x[s:s+self._clip_len, :],
                    "y": y,
                    "info": info
                })
        return rs
