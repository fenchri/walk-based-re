#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 08/09/2019

author: fenia
"""

import torch
from torch import nn
import numpy as np
from nnet.layers import *
from nnet.modules import EmbedLayer, Encoder, Classifier
np.set_printoptions(threshold=np.inf)
torch.set_printoptions(profile="full")


class BaseNet(nn.Module):
    def __init__(self, params, sizes, pembeds, lab2ign=None, o_type=None, maps=None):
        super(BaseNet, self).__init__()    

        self.w_dim = 2*params['lstm_dim']
        self.e_dim = 2*params['lstm_dim']

        self.maps = maps
        self.lab2ign = lab2ign
        self.o_type = o_type
        self.direction = params['direction']
        self.sizes = sizes
        self.device = torch.device("cuda:{}".format(params['gpu']) if params['gpu'] != -1 else "cpu")

        #############################################
        # Layers
        #############################################
        self.word_embed = EmbedLayer(num_embeddings=sizes['word_size'],
                                     embedding_dim=params['word_dim'],
                                     dropout=params['dropi'],
                                     ignore=None,
                                     freeze=False,
                                     pretrained=pembeds,
                                     mapping=maps['word2idx'])

        self.pos_embed = EmbedLayer(num_embeddings=sizes['pos_size']+1,
                                    embedding_dim=params['pos_dim'],
                                    dropout=0.0,
                                    ignore=sizes['pos_size'],
                                    freeze=False,
                                    pretrained=None,
                                    mapping=maps['pos2idx'])
        self.w_dim += 2 * params['pos_dim']
        self.e_dim += params['pos_dim']

        self.type_embed = EmbedLayer(num_embeddings=sizes['type_size']+1,
                                     embedding_dim=params['type_dim'],
                                     dropout=0.0,
                                     ignore=sizes['type_size'],
                                     freeze=False,
                                     pretrained=None,
                                     mapping=maps['type2idx'])
        self.w_dim += params['type_dim']
        self.e_dim += params['type_dim']

        self.encoder = Encoder(input_size=params['word_dim'],
                               rnn_size=params['out_dim'],
                               num_layers=1,
                               bidirectional=True,
                               dropout=0.0)

        if params['att']:
            self.attention = VectorAttentionLayer(self.w_dim, self.device)
            self.reduce = nn.Linear(2 * self.e_dim + self.w_dim, params['out_dim'], bias=False)
        else:
            self.reduce = nn.Linear(2 * self.e_dim, params['out_dim'], bias=False)

        if params['walks_iter'] > 0:
            self.walk_layer = WalkLayer(params['out_dim'], params['walks_iter'], params['beta'], self.device)

        self.classifier = Classifier(in_size=params['out_dim'],
                                     out_size=sizes['rel_size'],
                                     dropout=params['dropo'])

        #############################################
        # Hyper-parameters
        #############################################
        self.beta = params['beta']
        self.att = params['att']
        self.word_dim = params['word_dim']
        self.pos_dim = params['pos_dim']
        self.type_dim = params['type_dim']
        self.dropi = params['dropi']
        self.dropm = params['dropm']
        self.dropo = params['dropo']
        self.out_dim = params['out_dim']
        self.lstm_dim = params['lstm_dim']
        self.walks_iter = params['walks_iter']
        self.lr = params['lr']
        self.gc = params['gc']
        self.reg = params['reg']

