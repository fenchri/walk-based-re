#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

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
    np.random.seed(seed)              # Numpy module
    random.seed(seed)                 # Python random module
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(parameters):
    model_folder = setup_log(parameters, 'train')

    set_seed(0)

    ###################################
    # Data Loading
    ###################################
    print('\nLoading training data ...')
    train_loader = DataLoader(parameters['train_data'], parameters)
    train_loader(embeds=parameters['embeds'])
    train_data = RelationDataset(train_loader, 'train', parameters['unk_w_prob'], train_loader).__call__()

    print('\nLoading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters)
    test_loader()
    test_data = RelationDataset(test_loader, 'test', parameters['unk_w_prob'], train_loader).__call__()

    ###################################
    # TRAINING
    ###################################
    trainer = Trainer({'train': train_data, 'test': test_data}, parameters, train_loader, model_folder)
    trainer.run()

    trainer.eval_epoch(final=True, save_predictions=True)
    if parameters['plot']:
        plot_learning_curve(trainer, model_folder)


def test(parameters):
    print('*** Testing Model ***')
    model_folder = setup_log(parameters, 'test')

    print('Loading mappings ...')
    with open(os.path.join(model_folder, 'mappings.pkl'), 'rb') as f:
        loader = pkl.load(f)

    print('Loading testing data ...')
    test_loader = DataLoader(parameters['test_data'], parameters)
    test_loader.__call__()
    test_data = RelationDataset(test_loader, 'test', parameters['unk_w_prob'], loader).__call__()

    m = Trainer({'train': [], 'test': test_data}, parameters, loader, model_folder)
    trainer = load_model(model_folder, m)
    trainer.eval_epoch(final=True, save_predictions=True)


def main():
    config = ConfigLoader()
    parameters = config.load_config()

    if parameters['train']:
        train(parameters)

    elif parameters['test']:
        test(parameters)
    

if __name__ == "__main__":
    main()
