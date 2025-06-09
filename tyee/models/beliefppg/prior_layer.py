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
from scipy.stats import laplace, norm  


class TransitionPriorNotSetError(Exception):
    def __init__(self, message="Transition prior matrix not set, please fit the layer before calling"):
        self.message = message
        super().__init__(self.message)


class PriorLayer(nn.Module):
    """
    Belief propagation/decoding layer: converts raw heart rate probability distributions into context-aware predictions
    
    Features: 
    1. Online mode (sum-product): real-time belief state updates
    2. Batch mode (Viterbi): uses dynamic programming to find optimal paths
    3. Supports two uncertainty measures: information entropy or standard deviation
    
    Main application scenarios: 
    - Post-processing for heart rate prediction
    - Smoothing for time series signals
    - Context-aware probability distribution modeling
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
        Initialize parameters: 
        :param dim: number of frequency bins (i.e., dimension of probability distribution) 
        :param min_hz: minimum predictable frequency (Hz) 
        :param max_hz: maximum predictable frequency (Hz)  
        :param online: whether to use online belief propagation mode
        :param return_probs: whether to return processed probability distribution (otherwise return predictions) 
        :param transition_prior: predefined transition probability matrix (optional) 
        :param uncert: uncertainty measure method: "entropy" or "std" (standard deviation) 
        """
        super().__init__()
         
        self.register_buffer('state', torch.ones(dim) / dim) 
        
        self.dim = dim                      
        self.min_hz = min_hz               
        self.max_hz = max_hz               
        self.online = online                
        self.return_probs = return_probs    
        self.uncert = uncert                

        self.register_buffer(
            'bins',
            torch.tensor(
                [self._bin_hr_bpm(i) for i in range(dim)],
                dtype=torch.float32
            )
        )
        
        if transition_prior is not None:
            self.register_buffer('transition_prior', torch.tensor(transition_prior, dtype=torch.float32))
        else:
            self.transition_prior = None

    def _bin_hr_bpm(self, i: int) -> float:
        return self.min_hz * 60.0 + (self.max_hz - self.min_hz) * 60.0 * i / self.dim

    def _fit_distr(self, diffs, distr_type):
        """
        Fit probability distribution for heart rate changes
        :param diffs: sequence of heart rate change values 
        :param distr_type: distribution type: "laplace" or "gauss" 
        :return: (mean, standard deviation)
        """
        if distr_type == "laplace":
            return laplace.fit(diffs)  
        return norm.fit(diffs)         

    def fit_prior_layer(self, ys, distr_type="laplace", sparse=False):
        """
        Generate transition probability matrix from training data
        
        Processing steps: 
        1. Calculate consecutive heart rate differences
        2. Fit specified distribution
        3. Generate transition probability matrix
        
        :param ys: list of heart rate sequences (each element is a heart rate array for one time period) 
        :param distr_type: distribution type
        :param sparse: whether to apply sparsification (clip low-probability transitions) 
        """
        if distr_type == "laplace":
            diffs = np.concatenate([y[1:] - y[:-1] for y in ys])
        elif distr_type == "gauss":
            diffs = np.concatenate([np.log(y[1:]) - np.log(y[:-1]) for y in ys])
        else:
            raise NotImplementedError(f"Unsupported distribution type: {distr_type}")

        mu, sigma = self._fit_distr(diffs, distr_type)
        
        trans_prob = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            for j in range(self.dim):
                if sparse and abs(i - j) > 10 * 60 / self.dim:
                    continue  

                if distr_type == "laplace":
                    prob = laplace.cdf(abs(i-j)+1, mu, sigma) - laplace.cdf(abs(i-j)-1, mu, sigma)
                else:
                    log_diffs = [
                        np.log(self._bin_hr_bpm(i1)) - np.log(self._bin_hr_bpm(i2)) 
                        for i1 in (i-0.5, i+0.5) 
                        for i2 in (j-0.5, j+0.5)
                    ]
                    prob = norm.cdf(np.max(log_diffs), mu, sigma) - norm.cdf(np.min(log_diffs), mu, sigma)
                
                trans_prob[i][j] = prob
       
        self.transition_prior = torch.tensor(trans_prob, dtype=torch.float32)

    def _sum_product(self, ps):
        """
        Online belief propagation (sum-product message passing) 
        
        Processing steps: 
        1. Propagate previous state using transition matrix
        2. Multiply with current observation probability
        3. Normalize to get new state
        
        :param ps: probability distribution sequence for current time steps (seq_len, dim)
        :return: updated probability sequence (seq_len, dim)
        """
        if self.transition_prior is None:
            raise TransitionPriorNotSetError()

        self.transition_prior = self.transition_prior.to(ps.device)
        output = torch.zeros(size=(ps.shape[0], self.dim), dtype=ps.dtype, device=ps.device)
        for t in range(ps.size(0)):  
            # transition_prior (dim, dim) * state (dim,)
            p_prior = torch.mv(self.transition_prior, self.state)  
            
            p_new = p_prior * ps[t]  
            
            self.state = p_new / torch.sum(p_new)
            
            # outputs.append(self.state)
            output[t] = self.state
        
        return output

    def _decode_viterbi(self, raw_probs):
        """
        Viterbi decoding to find optimal path
        
        Implementation steps: 
        1. Forward pass: calculate maximum path probabilities
        2. Backward tracking: determine optimal path
        
        :param raw_probs: raw probability sequence (seq_len, dim)
        :return: predicted heart rate sequence (seq_len,)
        """
        seq_len, dim = raw_probs.shape
        device = raw_probs.device
        dim = self.transition_prior.size(0)
        self.transition_prior = self.transition_prior.to(device)
        paths = torch.zeros((seq_len, dim), dtype=torch.long, device=device)

        max_prob = torch.ones(dim, dtype=torch.float32, device=device) / dim  
        for t in range(raw_probs.size(0)):
            trans_prod = max_prob.unsqueeze(0) * self.transition_prior
            max_prob, indices = torch.max(trans_prod, dim=1)
            
            max_prob = max_prob * raw_probs[t]
            max_prob /= torch.sum(max_prob)  
            
            paths[t] = indices  

        best_path = torch.zeros(seq_len, dtype=torch.long, device=device)
        best_path[-1] = torch.argmax(max_prob)
        for t in reversed(range(1, len(paths))):
            best_path[t-1] = paths[t, best_path[t]]
        
        hr_bpm = self.bins[best_path]
        return hr_bpm

    def forward(self, probs):
        """
        Main forward propagation function
        
        Returns based on mode: 
        - Online mode: processed probability distribution + uncertainty
        - Batch mode: Viterbi decoding results
        
        :param probs: input probabilities (seq_len, dim)
        :return: prediction results and uncertainty (specific format depends on return_probs setting) 
        """
        if self.online:
            prior_probs = self._sum_product(probs)
            
            E_x = torch.sum(prior_probs * self.bins[None, :], dim=1)
            
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
            pred = self._decode_viterbi(probs)
            return pred, torch.zeros_like(pred)  
