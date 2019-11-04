#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 12:28:50 2018

author: fenia
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


class VectorAttentionLayer(nn.Module):
    """
    Compute attentive scores for each word in the sentence
    """
    def __init__(self, embed_size, device):
        super(VectorAttentionLayer, self).__init__() 
        self.q = nn.Parameter(nn.init.normal_(torch.empty(embed_size, 1)), requires_grad=True)  # (D,1)
        self.tanh = nn.Tanh()
        self.embed_size = embed_size
        self.device = device

    def forward(self, xs, mask=None):
        a_scores = torch.matmul(self.tanh(xs), self.q.unsqueeze(0))  # (BEE, W, 1)

        # replace with -inf so that softmax returns 0 in words that should not be included
        a_scores = torch.where(mask.unsqueeze(2), a_scores, torch.as_tensor([float('-inf')]).to(self.device))

        # if no context
        a_scores = torch.where(torch.isinf(a_scores).all(dim=1, keepdim=True),
                               torch.full_like(a_scores, 1.0), a_scores)
        a_scores = F.softmax(a_scores, dim=1)          # (BEE, W, 1)

        # if no context
        a_scores = torch.where(torch.eq(a_scores, 1/a_scores.shape[1]).all(dim=1, keepdim=True),
                               torch.zeros_like(a_scores), a_scores)

        y_expect = torch.matmul(a_scores.transpose(1, 2), xs)  # (BEE, 1, D)
        y_expect = torch.squeeze(y_expect, dim=1)
        return y_expect, torch.squeeze(a_scores, dim=2)


class WalkLayer(nn.Module):
    """
     Walks generation layer:
        1. Modified Bilinear
            to create a path between A and C through B --> AC * CB
        2. sigmoid
        3. sum pooling over all possible paths (intermediate nodes)
    """
    def __init__(self, embed_size, iter_, beta, device): 
        super(WalkLayer, self).__init__()       
        self.W = nn.Parameter(nn.init.normal_(torch.empty(embed_size, embed_size)), requires_grad=True)
        self.sigmoid = nn.Sigmoid()
        self.beta = beta
        self.embed_size = embed_size
        self.iter = iter_
        self.device = device

    def generate(self, old_pairs, part1, part2, items):
        # value == -1 indicates invalid pair (padded)
        bilin = torch.matmul(old_pairs, self.W)
        a = bilin[part1]
        b = old_pairs[part2]
        mask = (torch.ge(part1, 0) & torch.ge(part2, 0))  # mask padded intermediate entities
        
        a = torch.where(mask.unsqueeze(1), a, torch.zeros_like(a))
        b = torch.where(mask.unsqueeze(1), b, torch.zeros_like(b))
        a = a.view(-1, items, a.shape[1])
        b = b.view(-1, items, b.shape[1])

        new_pairs = a * b  # elementwise
        return old_pairs, new_pairs

    def aggregate(self, old_pairs, new_pairs, idx, items, map_pair):
        # mask invalid
        new_pairs[(idx[2] == idx[3]).view(-1, items)] = float('-inf')  # A -> C -> C
        new_pairs[(idx[1] == idx[3]).view(-1, items)] = float('-inf')  # A -> A -> C
        new_pairs[map_pair[:, torch.arange(items).to(self.device),
                              torch.arange(items).to(self.device)], :] = float('-inf')  # A -> * -> A
        new_pairs[torch.eq(new_pairs, torch.zeros_like(new_pairs)).all(dim=2)] = float('-inf')  # padded entities

        mat = torch.where(torch.isinf(new_pairs).all(dim=1),  # If no valid intermediate node
                          torch.ones_like(old_pairs),
                          torch.full_like(old_pairs, self.beta))  # [:, 0].unsqueeze(-1)

        new_pairs = torch.sum(self.sigmoid(new_pairs), dim=1)  # non-linearity & sum pooling
        # new_pairs = torch.lerp(new_pairs, old_pairs, weight=mat)  # interpolation
        new_pairs = (mat * old_pairs) + ((1 - mat) * new_pairs)  # interpolation
        return new_pairs

    def forward(self, pairs, cond, map_pair):
        cond, _ = torch.broadcast_tensors(cond.unsqueeze(-1), torch.zeros((1, 1, 1, cond.shape[1])))

        idx = torch.nonzero(cond).unbind(dim=1)  # batch [0], row [1], col [2], intermediate [3]
        items = map_pair.shape[1]

        part1 = map_pair[idx[0], idx[1], idx[3]]
        part2 = map_pair[idx[0], idx[3], idx[2]]

        for _ in range(0, self.iter):
            old_pairs, new_pairs = self.generate(pairs, part1, part2, items)
            pairs = self.aggregate(old_pairs, new_pairs, idx, items, map_pair)

        return pairs
