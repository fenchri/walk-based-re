#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 14:02:56 2018

author: fenia
"""

import random
import numpy as np
import itertools
from collections import OrderedDict
import argparse
import yaml
import yamlordereddictloader
from reader import read_relation_input
random.seed(0)
np.random.seed(0)


class ConfigLoader:
    def __init__(self):
        pass

    @staticmethod
    def load_cmd():
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', type=str, required=True, help='Yaml parameter file')
        parser.add_argument('--train', action='store_true', help='Training mode - model is saved')
        parser.add_argument('--test', action='store_true', help='Testing mode - needs a model to load')
        parser.add_argument('--gpu', type=int, required=True, help='GPU number, use -1 for CPU')
        parser.add_argument('--walks', type=int, help='Number of walk iterations')
        parser.add_argument('--att', type=str, help='Use attention or not', choices=['True', 'False'])
        parser.add_argument('--example', action='store_true', help='Print the sentences and info in the 1st batch, then exit (useful for debugging)')
        parser.add_argument('--direction', type=str, help='Direction of arguments to classify', choices=['l2r', 'r2l', 'l2r+r2l'])
        parser.add_argument('--folder', type=str, help='Destination folder to save model, predictions and errors')
        parser.add_argument('--embeds', type=str, help='Pre-trained word embeds file')
        parser.add_argument('--train_data', type=str, help='Training data file')
        parser.add_argument('--test_data', type=str, help='Test data dile')
        parser.add_argument('--epoch', type=int, help='Stopping epoch')
        parser.add_argument('--early_stop', action='store_true', help='Use early stopping')
        parser.add_argument('--preds', type=str, help='Folder name for predictions')
        return parser.parse_args()

    def load_config(self):
        inp = self.load_cmd()
        with open(vars(inp)['config'], 'r') as f:
            parameters = yaml.load(f, Loader=yamlordereddictloader.Loader)

        parameters = dict(parameters)
        parameters['train'] = inp.train
        parameters['test'] = inp.test
        parameters['gpu'] = inp.gpu
        parameters['example'] = inp.example

        if inp.att != None:
            parameters['att'] = inp.att

        if inp.walks:
            parameters['walks_iter'] = inp.walks

        if inp.folder:
            parameters['folder'] = inp.folder

        if inp.embeds:
            parameters['embeds'] = inp.embeds

        if inp.train_data:
            parameters['train_data'] = inp.train_data

        if inp.test_data:
            parameters['test_data'] = inp.test_data

        if inp.direction:
            parameters['direction'] = inp.direction

        if inp.epoch:
            parameters['epoch'] = inp.epoch

        if inp.preds:
            parameters['save_preds'] = inp.preds

        if inp.early_stop:
            parameters['early_stopping'] = inp.early_stop

        return parameters


class DataLoader:
    def __init__(self, input_file, params):
        self.input = input_file
        self.params = params

        self.max_distance = -9999999999
        self.min_distance = 9999999999
        self.embeds_file = params['embeds']
        self.pre_words = []
        self.pre_embeds = OrderedDict()
        self.max_distance = 0
        self.lower = params['lowercase']

        self.word2index, self.index2word, self.n_words, self.word2count = {'<UNK>': 0}, {0: '<UNK>'}, 1, {'<UNK>': 1}
        self.type2index, self.index2type, self.n_type, self.type2count = {'O': 0}, {0: 'O'}, 1, {'O': 0}
        self.rel2index, self.index2rel, self.n_rel, self.rel2count = {'1:NR:2': 0}, {0: '1:NR:2'}, 1, {'1:NR:2': 0}
        self.pos2index, self.index2pos, self.n_pos, self.pos2count = {'inside': 0, 'outside': 1}, \
                                                                     {0: 'inside', 1: 'outside'}, 2, \
                                                                     {'inside': 0, 'outside': 0}

        self.sentences, self.entities, self.pairs = OrderedDict(), OrderedDict(), OrderedDict()
        self.singletons = []
        self.label2ignore = 0
        self.reverse_l = []

    @staticmethod
    def normalize_string(string, str2rpl='0'):
        return re.sub("\d", str2rpl, string)

    def find_singletons(self, min_w_freq=1):
        """
        Find items with frequency <= 2 and based on probability
        """
        self.singletons = frozenset([elem for elem, val in self.word2count.items()
                                     if ((val <= min_w_freq) and elem != '<UNK>')])

    def add_relation(self, rel):
        if rel not in self.rel2index:
            self.rel2index[rel] = self.n_rel
            self.rel2count[rel] = 1
            self.index2rel[self.n_rel] = rel
            self.n_rel += 1
        else:
            self.rel2count[rel] += 1

    def add_word(self, word):
        if self.lower:
            word = word.lower()

        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_type(self, type):
        if type not in self.type2index:
            self.type2index[type] = self.n_type
            self.type2count[type] = 1
            self.index2type[self.n_type] = type
            self.n_type += 1
        else:
            self.type2count[type] += 1

    def add_pos(self, pos):
        pos = str(pos)
        if pos not in self.pos2index:
            self.pos2index[pos] = self.n_pos
            self.pos2count[pos] = 1
            self.index2pos[self.n_pos] = pos
            self.n_pos += 1
        else:
            self.pos2count[pos] += 1

    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)

    def find_maxmin_length(self, length):
        for l in length:
            if l-1 > self.max_distance:
                self.max_distance = l-1

            if -l+1 < self.min_distance:
                self.min_distance = -l+1

    def load_embeds(self, word_dim):
        """
        Load pre-trained word embeddings if specified
        """
        self.pre_embeds = OrderedDict()
        with open(self.embeds_file, 'r') as vectors:
            for x, line in enumerate(vectors):

                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]

                    n = len(vec)
                    if n != word_dim:
                        print('  Wrong dimensionality! -- line No{}, word: {}, len {}'.format(x, word, n))
                        continue
                    else:
                        self.add_word(word)
                        self.pre_embeds[word] = np.asarray(vec, 'f')
        self.pre_words = [w for w, e in self.pre_embeds.items()]
        print('  Found pre-trained word embeddings: {} x {}'.format(len(self.pre_embeds), word_dim), end="")

    def read_n_map(self):
        """
        Read input.
        """
        lengths, self.sentences, self.entities, self.pairs =\
            read_relation_input(self.input, self.sentences, self.entities, self.pairs)

        self.find_maxmin_length(lengths)

        # map types and positions and relation types
        for did, d in self.sentences.items():
            self.add_sentence(d)

        for did, e in self.entities.items():
            for k, v in e.items():
                self.add_type(v.type)

        for pos in np.arange(-self.max_distance, self.max_distance+1):
            self.add_pos(str(pos))

        for did, p in self.pairs.items():
            for k, v in p.items():
                self.add_relation(v.type)

        # make sure all directions are there
        current_rels = list(self.rel2index.keys())
        for k in current_rels:
            if k != '1:NR:2' and k != 'not_include':
               rev = k.split(':')
               if rev[2]+':'+rev[1]+':'+rev[0] not in current_rels:
                   print('relation not found -- adding', rev[2]+':'+rev[1]+':'+rev[0])
                   self.add_relation(rev[2]+':'+rev[1]+':'+rev[0])
                   self.rel2count[rev[2]+':'+rev[1]+':'+rev[0]] = 0

        assert len(self.entities) == len(self.sentences) == len(self.pairs)

    def reverse_labels(self):
        labmap = []
        for e in range(0, self.n_rel):
            x_ = self.index2rel[e].split(':')
            if x_[1] == 'NR':
                labmap += [self.rel2index['1:NR:2']]
            else:
                labmap += [self.rel2index[x_[2] + ':' + x_[1] + ':' + x_[0]]]
        labmap = np.array(labmap, 'i')
        return labmap

    def statistics(self):
        """ Print statistics for the dataset """
        print('  # Sentences: {:<5}\n  # words: {:<5}'.format(len(self.sentences), len(self.word2count.keys())))

        print('  # Relations: {}'.format(sum([v for k, v in self.rel2count.items()])))
        for k, v in sorted(self.rel2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.rel2index[k]))

        print('  # Entities: {}'.format(sum([len(e) for e in self.entities.values()])))
        for k, v in sorted(self.type2count.items()):
            print('\t{:<10}\t{:<5}\tID: {}'.format(k, v, self.type2index[k]))

        print('  # Singletons: {}/{}'.format(len(self.singletons), len(self.word2count.keys())))

    def __call__(self, embeds=None):
        self.read_n_map()
        self.find_singletons(self.params['min_w_freq'])  # words with freq=1
        self.reverse_l = self.reverse_labels()
        self.statistics()
        if embeds:
            self.load_embeds(self.params['word_dim'])
            print(' --> # Words + Pre-trained: {:<5}'.format(self.n_words))
