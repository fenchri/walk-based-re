#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 05/06/2019

author: fenia
"""

import random
random.seed(0)
import numpy as np
np.random.seed(0)
import re
import argparse
from collections import OrderedDict
from recordtype import recordtype
from tqdm import tqdm
from torch.utils.data import Dataset
import itertools
from loader import DataLoader


class RelationDataset:
    """
    A constructor for a relation dataset.
    """
    def __init__(self, loader, data_type, unk_w_prob, mappings):
        self.unk_w_prob = unk_w_prob
        self.mappings = mappings
        self.loader = loader
        self.data_type = data_type
        self.data = []

    def __call__(self):
        pbar = tqdm(self.loader.sentences.keys())

        all_l2r = 0
        for pmid in pbar:
            pbar.set_description('  Preparing {} data - Sentence ID {}'.format(self.data_type.upper(), pmid))

            # TEXT
            if self.data_type == 'train':
                sent = []
                for w, word in enumerate(self.loader.sentences[pmid]):
                    if self.loader.lower:
                        word = word.lower()  # make lowercase

                    if (word in self.mappings.singletons) and (random.uniform(0, 1) < float(self.unk_w_prob)):
                        sent += [self.mappings.word2index['<UNK>']]  # UNK words = singletons
                    else:
                        sent += [self.mappings.word2index[word]]

            else:
                sent = []
                for w, word in enumerate(self.loader.sentences[pmid]):
                    if self.loader.lower:
                        word = word.lower()  # make lowercase

                    if word in self.mappings.word2index:
                        sent += [self.mappings.word2index[word]]
                    else:
                        sent += [self.mappings.word2index['<UNK>']]
            assert len(self.loader.sentences[pmid]) == len(sent), '{}, {}'.format(len(sentence), len(sent))
            sent = np.array(sent, 'i')

            # ENTITIES [id, type, start, end]
            ent = []
            for id_, e in enumerate(self.loader.entities[pmid].values()):
                if e.type not in self.mappings.type2index:
                    # print('Entity type not found', e.type)
                    ent += [[id_, -1, int(e.start), int(e.end)]]
                else:
                    ent += [[id_, self.mappings.type2index[e.type], int(e.start), int(e.end)]]
            ent = np.array(ent, 'i')

            # RELATIONS
            ents_keys = list(self.loader.entities[pmid].keys())  # in order
            true_rels = -1 * np.ones((len(ents_keys), len(ents_keys)), 'i')
            rel_info = np.empty((len(ents_keys), len(ents_keys)), dtype='object_')
            l2r = -1 * np.ones((len(ents_keys), len(ents_keys)), 'i')
            for id_, r in enumerate(self.loader.pairs[pmid].values()):

                true_rels[ents_keys.index(r.arg1), ents_keys.index(r.arg2)] = self.mappings.rel2index[r.type]
                rel_info[ents_keys.index(r.arg1),
                         ents_keys.index(r.arg2)] = {'pmid': pmid,
                                                     'doc': self.loader.sentences[pmid],
                                                     'entA': self.loader.entities[pmid][r.arg1],
                                                     'entB': self.loader.entities[pmid][r.arg2],
                                                     'rel': self.mappings.rel2index[r.type],
                                                     'dir': r.direction}

                # inverse
                if r.type != '1:NR:2':
                    r_type_ = self.mappings.rel2index[r.type.split(':')[2] + ':' +
                                                      r.type.split(':')[1] + ':' +
                                                      r.type.split(':')[0]]
                    true_rels[ents_keys.index(r.arg2), ents_keys.index(r.arg1)] = r_type_
                    rel_info[ents_keys.index(r.arg2),
                             ents_keys.index(r.arg1)] = {'pmid': pmid,
                                                         'doc': self.loader.sentences[pmid],
                                                         'entA': self.loader.entities[pmid][r.arg2],
                                                         'entB': self.loader.entities[pmid][r.arg1],
                                                         'rel': r_type_,
                                                         'dir': r.direction}
                else:
                    true_rels[ents_keys.index(r.arg2), ents_keys.index(r.arg1)] = self.mappings.rel2index[r.type]
                    rel_info[ents_keys.index(r.arg2),
                             ents_keys.index(r.arg1)] = {'pmid': pmid,
                                                         'doc': self.loader.sentences[pmid],
                                                         'entA': self.loader.entities[pmid][r.arg2],
                                                         'entB': self.loader.entities[pmid][r.arg1],
                                                         'rel': self.mappings.rel2index[r.type],
                                                         'dir': r.direction}

                if int(self.loader.entities[pmid][r.arg1].end)-1 <= int(self.loader.entities[pmid][r.arg2].start):
                    l2r[ents_keys.index(r.arg1), ents_keys.index(r.arg2)] = 1
                else:
                    l2r[ents_keys.index(r.arg2), ents_keys.index(r.arg1)] = 1

            all_l2r += np.sum(l2r != -1)

            # POSITIONS
            # Create edge distances

            # entity-entity dist
            xv, yv = np.meshgrid(np.arange(ent.shape[0]), np.arange(ent.shape[0]), indexing='ij')
            dist_ee = np.empty((ent.shape[0], ent.shape[0]), 'i')

            a_start, a_end = ent[xv, 2], ent[xv, 3]-1
            b_start, b_end = ent[yv, 2], ent[yv, 3]-1

            dist_ee = np.where((a_end < b_start) & (b_start != -1) & (a_end != -1), b_start - a_end, dist_ee)
            dist_ee = np.where((b_end < a_start) & (b_end != -1) & (a_start != -1), b_end - a_start, dist_ee)

            # limit max distance according to training set
            dist_ee = np.where(dist_ee > self.mappings.max_distance, self.mappings.max_distance, dist_ee)
            dist_ee = np.where(dist_ee < self.mappings.min_distance, self.mappings.min_distance, dist_ee)

            dist_ee = np.where((b_start <= a_start) & (b_end >= a_end)  # a is inside
                               & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), 'inside', dist_ee)
            dist_ee = np.where((b_start >= a_start) & (b_end <= a_end)  # a is outside
                               & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), 'outside', dist_ee)
            dist_ee[np.arange(ent.shape[0]), np.arange(ent.shape[0])] = '0'  # diagonal to zero

            dist_ee = list(map(lambda y: self.mappings.pos2index[y], dist_ee.ravel().tolist()))  # map
            dist_ee = np.array(dist_ee, 'i').reshape((ent.shape[0], ent.shape[0]))

            # entity-token dist
            xz, yz = np.meshgrid(np.arange(ent.shape[0]), np.arange(len(self.loader.sentences[pmid])), indexing='ij')
            tokens = np.arange(len(self.loader.sentences[pmid]))

            dist_et = np.empty((ent.shape[0], len(self.loader.sentences[pmid])), 'i')
            a_start, a_end = ent[xz, 2], ent[xz, 3]-1
            b_start, b_end = tokens[yz], tokens[yz]

            dist_et = np.where((a_end < b_start) & (b_start != -1) & (a_end != -1), b_start - a_end, dist_et)
            dist_et = np.where((b_end < a_start) & (b_end != -1) & (a_start != -1), b_end - a_start, dist_et)

            # limit max distance according to training set
            dist_et = np.where(dist_et > self.mappings.max_distance, self.mappings.max_distance, dist_et)
            dist_et = np.where(dist_et < self.mappings.min_distance, self.mappings.min_distance, dist_et)

            dist_et = np.where((b_start <= a_start) & (b_end >= a_end)
                               & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), '0', dist_et)
            dist_et = np.where((b_start >= a_start) & (b_end <= a_end)
                               & (b_start != -1) & (a_end != -1) & (b_end != -1) & (a_start != -1), '0', dist_et)

            dist_et = list(map(lambda y: self.mappings.pos2index[y], dist_et.ravel().tolist()))
            dist_et = np.array(dist_et, 'i').reshape((ent.shape[0], len(self.loader.sentences[pmid])))

            self.data += [OrderedDict({'sentId': pmid, 'text': sent, 'ents': ent, 'rels': true_rels,
                                       'pos_ee': dist_ee, 'pos_et': dist_et, 'info': rel_info,
                                       'word': np.array(len(self.loader.sentences[pmid]), 'i'),
                                       'entity': np.array(ent.shape[0], 'i'),
                                       'l2r': l2r})]
        assert all_l2r == sum([v for k, v in self.loader.rel2count.items()]), \
            '{} <> {}'.format(all_l2r, sum([v for k, v in self.loader.rel2count.items()]))

        return self.data

    def __len__(self):
        return len(self.data)


