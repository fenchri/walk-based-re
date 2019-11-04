#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 26/09/2019

author: fenia
"""

import torch
import os
import random
import numpy as np
from dataset import RelationDataset
from loader import DataLoader, ConfigLoader
from nnet.trainer import Trainer
from utils import ordered_load, setup_log, save_model, load_model, plot_learning_curve
import pickle as pkl

torch.set_printoptions(profile="full")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class ModelWrapper:
    def __init__(self):
        config = ConfigLoader()
        self.parameters = config.load_config()
        self.model_folder = setup_log(self.parameters, 'train')

        set_seed(0)

        ###################################
        # Data Loading
        ###################################
        print('\nLoading training data ...')
        self.train_loader = DataLoader(parameters['train_data'], self.parameters)
        train_loader(embeds=self.parameters['embeds'])
        self.train_data = RelationDataset(train_loader, 'train', self.parameters['unk_w_prob'], train_loader).__call__()

        print('\nLoading testing data ...')
        test_loader = DataLoader(self.parameters['test_data'], parameters)
        test_loader()
        self.test_data = RelationDataset(test_loader, 'test', self.parameters['unk_w_prob'], train_loader).__call__()

    def update_hyperparams(self, params2upd):
        for item in params2upd.keys():
            self.parameters[item] = params2upd[item]

    def train(self):
        ###################################
        # TRAINING
        ###################################
        trainer = Trainer({'train': self.train_data, 'test': self.test_data},
                          self.parameters, self.train_loader, self.model_folder)
        trainer.run()

        trainer.eval_epoch(final=True, save_predictions=True)
        save_model(model_folder, trainer.model, train_loader)
        if parameters['plot']:
            plot_learning_curve(trainer, model_folder)

        return float(trainer.best_score)
