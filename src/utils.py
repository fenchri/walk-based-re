#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

@author: fenia
"""

import os
import yaml
import sys
from collections import OrderedDict
import logging
import torch
import pickle as pkl
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def setup_log(params, mode):
    if params['walks_iter'] == 0:
        length = 1
    else:
        length = 2**params['walks_iter']
    folder_name = 'beta{}-walks{}-att_{}-dir_{}'.format(
        params['beta'],
        length, params['att'], params['direction'])

    model_folder = os.path.join(params['folder'], folder_name)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    log_file = os.path.join(model_folder, 'info_'+mode+'.log')

    f = open(log_file, 'w')
    sys.stdout = Tee(sys.stdout, f)
    return model_folder
    

class Tee(object):
    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f_ in self.files:
            f_.write(obj)
            f_.flush()  # If you want the output to be visible immediately

    def flush(self):
        for f_ in self.files:
            f_.flush()


def ordered_load(stream, loader=yaml.Loader, object_pairs_hook=OrderedDict):
    """
    Load yaml parameters in order
    """
    class OrderedLoader(loader):
        pass

    def construct_mapping(loader_, node):
        loader.flatten_mapping(node)
        return object_pairs_hook(loader_.construct_pairs(node))

    OrderedLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        construct_mapping)

    return yaml.load(stream, OrderedLoader)


def humanized_time(second):
    """
    Args:
        second: time in seconds
    Returns: human readable time (hours, minutes, seconds)
    """
    m, s = divmod(second, 60)
    h, m = divmod(m, 60)
    return "%dh %02dm %02ds" % (h, m, s)


def observe(model):
    """
    Observe model parameters: name, range of matrices & gradients

    Args:
        model: specified model object
    """
    for p_name, param in model.namedparams():
        p_data, p_grad = param.data, param.grad
        print('Name: %s, Range of data: [%f, %f], Range of gradient: [%f, %f]' %
              (p_name, np.min(chainer.cuda.to_cpu(p_data.data)), 
               np.max(chainer.cuda.to_cpu(p_data.data)), 
               np.min(chainer.cuda.to_cpu(p_grad.data)), 
               np.max(chainer.cuda.to_cpu(p_grad.data))))


def plot_learning_curve(trainer, model_folder):
    """
    Plot the learning curves for training and test set (loss and primary score measure)

    Args:
        trainer (Class): trainer object
        model_folder (str): folder to save figures
    """
    print('Plotting learning curves ... ', end="")
    x = list(map(int, np.arange(len(trainer.train_res['loss']))))
    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x, trainer.train_res['loss'], 'b', label='train')
    plt.plot(x, trainer.test_res['loss'], 'g', label='test')
    plt.legend()
    plt.ylabel('Loss')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(x)

    plt.subplot(2, 1, 2)
    plt.plot(x, trainer.train_res['score'], 'b', label='train')
    plt.plot(x, trainer.test_res['score'], 'g', label='test')
    plt.legend()
    plt.ylabel('F1-score')
    plt.xlabel('Epochs')
    plt.yticks(np.arange(0, 1, 0.1))
    plt.xticks(x)

    fig.savefig(model_folder + '/learn_curves.png', bbox_inches='tight')
    print('END')


def print_hyperparams(model_0):
    print("""\nModel hyper-parameters:
                - learn   {}
                - reg     {}
                - dropi   {}
                - dropo   {}
                - type    {}
                - pos     {}
                - gradc   {} 
                - out_dim {}
                - beta    {} """.format(model_0.lr,
                                        model_0.reg,
                                        model_0.dropi,
                                        model_0.dropo,
                                        model_0.type_dim,
                                        model_0.pos_dim,
                                        model_0.gc,
                                        model_0.out_dim,
                                        model_0.beta))


def print_options(model_0, parameters):
    print("""\nModel options:
             - Train Data  {}
             - Test Data   {}
             - Embeddings  {}
             - Save folder {}
             - batchsize   {}

             - walks_iter   {} --> Length = {}
             - att          {}
             - param_avg    {}
             - nested       {}
             - early_metric {}
             - direction    {}
             - lowercase    {}\n""".format(parameters['train_data'], parameters['test_data'],
                                           parameters['embeds'], parameters['folder'],
                                           parameters['batch'],
                                           model_0.walks_iter, 2 ** parameters['walks_iter'],
                                           model_0.att,
                                           parameters['param_avg'],
                                           parameters['nested'],
                                           parameters['early_metric'],
                                           parameters['direction'],
                                           parameters['lowercase']))


def save_model(model_folder, model_0, loader):
    print('Saving the model ... ', end="")
    with open(os.path.join(model_folder, 'mappings.pkl'), 'wb') as f:
        pkl.dump(loader, f, pkl.HIGHEST_PROTOCOL)
    torch.save(model_0.state_dict(), os.path.join(model_folder, 're.model'))
    print('END')


def load_model(model_folder, m):
    print('\nLoading model ... ', end="")
    m.model.load_state_dict(torch.load(os.path.join(model_folder, 're.model'), map_location=m.model.device))
    print('END')
    return m


def print_results(scores, show_class, time):
    def indent(txt, spaces=18):
        return "\n".join(" " * spaces + ln for ln in txt.splitlines())

    if show_class:
        # print results for every class
        scores['per_class'].append(['-----', None, None, None])
        scores['per_class'].append(['macro score', scores['macro_p'], scores['macro_r'], scores['macro_f']])
        scores['per_class'].append(['micro score', scores['micro_p'], scores['micro_r'], scores['micro_f']])
        print(' | Elapsed time: {}\n'.format(humanized_time(time)))
        print(indent(tabulate(scores['per_class'],
                              headers=['Class', 'P', 'R', 'F1'],
                              tablefmt='orgtbl',
                              floatfmt=".4f",
                              missingval="")))
        print()
    else:
        # print overall scores
        print(' | MICRO P/R/F1 = {:.04f}\t{:.04f}\t{:.04f} | '
              'MACRO P/R/F1 = {:.04f}\t{:.04f}\t{:.04f} | '.format(scores['micro_p'], scores['micro_r'],
                                                                   scores['micro_f'], scores['macro_p'],
                                                                   scores['macro_r'], scores['macro_f']), end="")

        print('TP/ACTUAL/PRED {:<6}/{:<6}/{:<6} TOTAL {}'.format(scores['tp'], scores['actual'], scores['pred'],
                                                                  scores['total']), end="")
        print(' | {}'.format(humanized_time(time)))


def write_pred2file(predicts, probabs, rels_info, savef, rel_map):
    """
    Write predictions to specific file in 'savef' folder
    Args:
        predicts: predictions
        rels_info: gold relations information
        savef: save folder
        rel_map: mapping of relation types
    """
    print('Writing predictions to file ... ', end="")
    if not os.path.exists(savef):
        os.makedirs(savef)

    assert len(predicts) == len(rels_info) == len(probabs), \
        '{} predictions != {} relations != {} probabilities'.format(len(predicts), len(rels_info), len(probabs))

    with open(os.path.join(savef, 'preds.txt'), 'w') as outfile:
        for pred, prob, pair_info in zip(predicts, probabs, rels_info):
            doc_id = pair_info['pmid']
            arg1 = pair_info['entA']
            arg2 = pair_info['entB']

            prediction = rel_map[int(pred)]
            outfile.write('{}|{}|{}|{}|{}\n'.format(doc_id, arg1.id, arg2.id, prediction, prob))
    print('END')


def write_errors2file(predicts, rels_info, savef, map_=None):
    """ Write model errors to file """
    print('Writing errors to file ...', end="")
    if not os.path.exists(savef):
        os.makedirs(savef)

    assert len(predicts) == len(rels_info), '{} predictions != {} relations'.format(len(predicts), len(rels_info))

    with open(os.path.join(savef, 'errors.txt'), 'w') as outfile:
        for pred, pair_info in zip(predicts, rels_info):

            doc_id = pair_info['pmid']
            arg1 = pair_info['entA']
            arg2 = pair_info['entB']

            prediction = map_[int(pred)]
            truth = map_[int(pair_info['rel'])]

            if prediction != truth:
                outfile.write('Prediction --> {} \t Truth --> {}\n'.format(prediction, truth))
                outfile.write('DocID: {}\n{}\n'.format(doc_id, ' '.join(pair_info['doc'])))
                outfile.write('Arg1: {} ({})\ttokens: {}-{}\n'.format(arg1.name, arg1.type, arg1.start, int(arg1.end)-1))
                outfile.write('Arg2: {} ({})\ttokens: {}-{}\n'.format(arg2.name, arg2.type, arg2.start, int(arg2.end)-1))
                outfile.write('\n')
    print('END')


def write_bingo2file(predicts, rels_info, savef, map_=None):
    """ Write correct predictions to file """
    print('Writing correct predictions to file ...', end="")
    if not os.path.exists(savef):
        os.makedirs(savef)

    assert len(predicts) == len(rels_info), '{} predictions != {} relations'.format(len(predicts), len(rels_info))

    with open(os.path.join(savef, 'correct.txt'), 'w') as outfile:
        for pred, pair_info in zip(predicts, rels_info):

            doc_id = pair_info['pmid']
            arg1 = pair_info['entA']
            arg2 = pair_info['entB']

            prediction = map_[int(pred)]
            truth = map_[int(pair_info['rel'])]

            if prediction == truth and truth != '1:NR:2':   # write only the positives
                outfile.write('Prediction --> {} \t Truth --> {}\n'.format(prediction, truth))
                outfile.write('DocID: {}\n{}\n'.format(doc_id, ' '.join(pair_info['doc'])))
                outfile.write('Arg1: {} ({})\ttokens: {}-{}\n'.format(arg1.name, arg1.type, arg1.start, int(arg1.end)-1))
                outfile.write('Arg2: {} ({})\ttokens: {}-{}\n'.format(arg2.name, arg2.type, arg2.start, int(arg2.end)-1))
                outfile.write('\n')
    print('END')
