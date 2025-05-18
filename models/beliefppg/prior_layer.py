#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@Author  : zhoutao
@License : (C) Copyright 2016-2025, Hunan University
@Contact : zhoutau@outlook.com
@Software: Visual Studio Code
@File    : prior_layer.py
@Time    : 2025/03/30 15:05:17
@Desc    : 
"""
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import laplace, norm  # 用于概率分布拟合


# 自定义异常: 当未设置转移先验矩阵时抛出
class TransitionPriorNotSetError(Exception):
    def __init__(self, message="未设置转移先验矩阵，调用前请先拟合该层"):
        self.message = message
        super().__init__(self.message)


class PriorLayer(nn.Module):
    """
    信念传播/解码层: 将原始心率概率分布转换为上下文感知的预测
    
    功能说明: 
    1. 在线模式 (sum-product): 实时更新信念状态
    2. 批量模式 (Viterbi): 使用动态规划寻找最优路径
    3. 支持两种不确定性度量方式: 信息熵或标准差
    
    主要应用场景: 
    - 心率预测的后处理
    - 时序信号的平滑处理
    - 概率分布的上下文关联
    """
    def __init__(
            self,
            dim,
            min_hz,
            max_hz,
            online,
            return_probs,
            transition_prior=None,
            uncert="entropy"
        ) -> None:
        """
        初始化参数: 
        :param dim: 频率分箱数量 (即概率分布的维度) 
        :param min_hz: 最小可预测频率 (Hz) 
        :param max_hz: 最大可预测频率 (Hz)  
        :param is_online: 是否使用在线信念传播模式
        :param return_probs: 是否返回处理后的概率分布 (否则返回预测值) 
        :param transition_prior: 预定义的转移概率矩阵 (可选) 
        :param uncert: 不确定性度量方式: "entropy" (熵) 或 "std" (标准差) 
        """
        super().__init__()
        
        # 注册持久化状态 (不参与梯度计算) 
        self.register_buffer('state', torch.ones(dim) / dim)  # 初始化为均匀分布
        
        # 基础参数设置
        self.dim = dim                      # 分箱数量
        self.min_hz = min_hz                # 最小频率 (转换为BPM需要乘以60) 
        self.max_hz = max_hz                # 最大频率
        self.online = online                # 运行模式开关
        self.return_probs = return_probs    # 输出类型控制
        self.uncert = uncert                # 不确定性计算方式

        # 初始化频率分箱值 (BPM单位) 
        self.register_buffer(
            'bins',
            torch.tensor(
                [self._bin_hr_bpm(i) for i in range(dim)],
                dtype=torch.float32
            )
        )
        
        # 转移概率矩阵初始化
        if transition_prior is not None:
            self.register_buffer('transition_prior', torch.tensor(transition_prior, dtype=torch.float32))
        else:
            self.transition_prior = None

    def _bin_hr_bpm(self, i: int) -> float:
        """将分箱索引转换为实际心率值 (BPM) """
        return self.min_hz * 60.0 + (self.max_hz - self.min_hz) * 60.0 * i / self.dim

    def _fit_distr(self, diffs, distr_type):
        """
        拟合心率变化的概率分布
        :param diffs: 心率变化量序列 
        :param distr_type: 分布类型: "laplace" (拉普拉斯) 或 "gauss" (高斯) 
        :return: (均值, 标准差)
        """
        if distr_type == "laplace":
            return laplace.fit(diffs)  # 拟合拉普拉斯分布
        return norm.fit(diffs)         # 拟合高斯分布

    def fit_prior_layer(self, ys, distr_type="laplace", sparse=False):
        """
        根据训练数据生成转移概率矩阵
        
        处理流程: 
        1. 计算连续心率差值
        2. 拟合指定分布
        3. 生成转移概率矩阵
        
        :param ys: 心率序列列表 (每个元素为一个时间段的心率数组) 
        :param distr_type: 分布类型
        :param sparse: 是否进行稀疏化处理 (裁剪小概率转移) 
        """
        # 计算心率变化差值
        if distr_type == "laplace":
            diffs = np.concatenate([y[1:] - y[:-1] for y in ys])
        elif distr_type == "gauss":
            diffs = np.concatenate([np.log(y[1:]) - np.log(y[:-1]) for y in ys])
        else:
            raise NotImplementedError(f"不支持的分布类型: {distr_type}")

        # 拟合分布参数
        mu, sigma = self._fit_distr(diffs, distr_type)
        
        # 生成转移概率矩阵
        trans_prob = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                # 稀疏处理: 裁剪过大跳变
                if sparse and abs(i - j) > 10 * 60 / self.dim:
                    continue  
                
                # 计算转移概率
                if distr_type == "laplace":
                    # 拉普拉斯分布的概率密度积分
                    prob = laplace.cdf(abs(i-j)+1, mu, sigma) - laplace.cdf(abs(i-j)-1, mu, sigma)
                else:
                    # 高斯分布的对数差值积分
                    log_diffs = [
                        np.log(self._bin_hr_bpm(i1)) - np.log(self._bin_hr_bpm(i2)) 
                        for i1 in (i-0.5, i+0.5) 
                        for i2 in (j-0.5, j+0.5)
                    ]
                    prob = norm.cdf(np.max(log_diffs), mu, sigma) - norm.cdf(np.min(log_diffs), mu, sigma)
                
                trans_prob[i][j] = prob

        # 转换为PyTorch张量并注册        
        self.transition_prior = torch.tensor(trans_prob, dtype=torch.float32)

    def _sum_product(self, ps):
        """
        在线信念传播 (sum-product消息传递) 
        
        处理步骤: 
        1. 使用转移矩阵传播上一状态
        2. 与当前观测概率相乘
        3. 归一化得到新状态
        
        :param ps: 当前时间步的概率分布序列 (seq_len, dim)
        :return: 更新后的概率序列 (seq_len, dim)
        """
        if self.transition_prior is None:
            raise TransitionPriorNotSetError()

        self.transition_prior = self.transition_prior.to(ps.device)
        output = torch.zeros(size=(ps.shape[0], self.dim), dtype=ps.dtype, device=ps.device)
        for t in range(ps.size(0)):  # 遍历每个时间步
            # 状态传播: transition_prior (dim, dim) * state (dim,)
            p_prior = torch.mv(self.transition_prior, self.state)  
            
            # 与当前观测融合
            p_new = p_prior * ps[t]  
            
            # 归一化
            self.state = p_new / torch.sum(p_new)
            
            # outputs.append(self.state)
            output[t] = self.state
        
        return output

    def _decode_viterbi(self, raw_probs):
        """
        维特比解码寻找最优路径
        
        实现步骤: 
        1. 前向传递: 计算最大路径概率
        2. 反向追踪: 确定最优路径
        
        :param raw_probs: 原始概率序列 (seq_len, dim)
        :return: 预测心率序列 (seq_len,)
        """
        seq_len, dim = raw_probs.shape
        device = raw_probs.device
        dim = self.transition_prior.size(0)
        self.transition_prior = self.transition_prior.to(device)
        paths = torch.zeros((seq_len, dim), dtype=torch.long, device=device)

        max_prob = torch.ones(dim, dtype=torch.float32, device=device) / dim  # 初始化为均匀分布
        # 前向传递: 记录最优路径
        for t in range(raw_probs.size(0)):
            # 计算转移概率并找到最大值
            trans_prod = max_prob.unsqueeze(0) * self.transition_prior
            max_prob, indices = torch.max(trans_prod, dim=1)
            
            # 更新路径概率
            max_prob = max_prob * raw_probs[t]
            max_prob /= torch.sum(max_prob)  # 数值稳定
            
            paths[t] = indices  # 记录路径索引

        # 反向追踪: 从最终最大概率点回溯
        best_path = torch.zeros(seq_len, dtype=torch.long, device=device)
        best_path[-1] = torch.argmax(max_prob)
        for t in reversed(range(1, len(paths))):
            best_path[t-1] = paths[t, best_path[t]]
        
        hr_bpm = self.bins[best_path]
        # 转换为实际心率值
        return hr_bpm

    def forward(self, probs):
        """
        前向传播主函数
        
        根据模式返回: 
        - 在线模式: 处理后的概率分布 + 不确定性
        - 批量模式: 维特比解码结果
        
        :param probs: 输入概率 (seq_len, dim)
        :return: 预测结果与不确定性 (具体形式取决于return_probs设置) 
        """
        if self.online:
            # 在线信念传播模式
            prior_probs = self._sum_product(probs)
            
            # 计算期望值
            E_x = torch.sum(prior_probs * self.bins[None, :], dim=1)
            
            # 计算不确定性
            if self.uncert == "entropy":
                uncertainty = -torch.sum(prior_probs * torch.log(prior_probs + 1e-10), dim=1)
            else:
                E_x2 = torch.sum(prior_probs * (self.bins[None, :] ** 2), dim=1)
                uncertainty = torch.sqrt(E_x2 - E_x ** 2)
            
            if self.return_probs:
                return (prior_probs, uncertainty)
            else:
                return (E_x, uncertainty)
        else:
            # 维特比解码模式
            pred = self._decode_viterbi(probs)
            return pred, torch.zeros_like(pred)  # 不确定性暂不支持
