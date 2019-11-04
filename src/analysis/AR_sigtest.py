#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 15/08/2019

author: fenia
"""

import argparse
from collections import OrderedDict
from tqdm import tqdm
import random
import numpy as np


def load_data(args):
    all, intra, inter = {}, {}, {}
    all['A'], all['B'], all['true'] = {}, {}, {}

    for system, typ in zip([args.systemA, args.systemB, args.truth], ['A', 'B', 'true']):
        with open(system, 'r') as pred:
            for line in pred:
                line = line.rstrip().split('|')

                # format: {(PMID, arg1, arg2): rlabel}
                if ':' in line[3]:
                    doc_id = line[0].split('**')[0].split('.split')[0]
                    arg1 = line[1]
                    arg2 = line[2]
                    label = line[3]
                else:
                    doc_id = line[0].split('**')[0].split('.split')[0]
                    arg1 = line[1]
                    arg2 = line[2]
                    label = '1:'+line[3]+':2'

                if label != '1:NR:2':
                    all[typ].update({(doc_id, arg1, arg2): label})
    return all


def align(all):
    all_n = {'A': OrderedDict(), 'B': OrderedDict(), 'true': OrderedDict()}

    print(len(all['A']), len(all['B']), len(all['true']))

    for key in list(all['A'].keys()) + list(all['B'].keys()) + list(all['true'].keys()):

        if key in all_n['A'] or (key[0], key[2], key[1]) in all_n['A']:
            continue
        elif key in all['A']:
            all_n['A'][key] = all['A'][key]
        elif (key[0], key[2], key[1]) in all['A']:
            t = all['A'][(key[0], key[2], key[1])].split(':')
            all_n['A'][key] = t[2]+':'+t[1]+':'+t[0]
        else:
            all_n['A'][key] = '1:NR:2'

        if key in all_n['B'] or (key[0], key[2], key[1]) in all_n['B']:
            continue
        elif key in all['B']:
            all_n['B'][key] = all['B'][key]
        elif (key[0], key[2], key[1]) in all['B']:
            t = all['B'][(key[0], key[2], key[1])].split(':')
            all_n['B'][key] = t[2]+':'+t[1]+':'+t[0]
        else:
            all_n['B'][key] = '1:NR:2'

        if key in all_n['true'] or (key[0], key[2], key[1]) in all_n['true']:
            continue
        elif key in all['true']:
            all_n['true'][key] = all['true'][key]
        elif (key[0], key[2], key[1]) in all['true']:
            t = all['true'][(key[0], key[2], key[1])].split(':')
            all_n['true'][key] = t[2]+':'+t[1]+':'+t[0]
        else:
            all_n['true'][key] = '1:NR:2'

    return all_n


def eval_(t, y):
    t = list(t.values())
    y = list(y.values())

    label_num = len(map_)
    ignore_label = 0

    mask_t = np.equal(t, ignore_label)  # where the ground truth needs to be ignored
    mask_p = np.equal(y, ignore_label)  # where the predicted needs to be ignored

    true = np.where(mask_t, label_num, t)  # ground truth
    pred = np.where(mask_p, label_num, y)  # output of NN

    tp_mask = np.where(np.equal(pred, true), true, label_num)
    fp_mask = np.where(np.not_equal(pred, true), pred, label_num)
    fn_mask = np.where(np.not_equal(pred, true), true, label_num)

    tp = np.sum(np.bincount(tp_mask, minlength=label_num+1)[:label_num])
    fp = np.sum(np.bincount(fp_mask, minlength=label_num+1)[:label_num])
    fn = np.sum(np.bincount(fn_mask, minlength=label_num+1)[:label_num])
    return prf(tp, fp, fn)


def prf(tp, fp, fn):
    micro_r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    micro_p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    micro_f = ((2 * micro_p * micro_r) / (micro_p + micro_r)) if micro_p != 0.0 and micro_r != 0.0 else 0.0
    return {'p': micro_p, 'r': micro_r, 'f': micro_f}
    # return micro_f


def sig_test(args, system_A, system_B, truth):
    """
    Approximate Randomization significance test
    https://cs.stanford.edu/people/wmorgan/sigtest.pdf
    """
    r = 0
    for R_ in tqdm(range(0, args.R)):
        listX = OrderedDict()
        listY = OrderedDict()
        k = 0

        for d in system_A.keys():
            choose = random.randint(0, 1)
            if choose == 0:
                listX[d] = system_A[d]
                listY[d] = system_B[d]
            else:
                listX[d] = system_B[d]
                listY[d] = system_A[d]

        t_xy = np.abs(eval_(listX, truth)['f'] - eval_(listY, truth)['f'])
        t_ab = np.abs(eval_(system_A, truth)['f'] - eval_(system_B, truth)['f'])

        if t_xy >= t_ab:
            r += 1

    significance = (r+1)/(args.R+1)
    if significance < 0.05:
        decision = 'SIG !!! :D'
    else:
        decision = 'NOT SIG :('
    print('Significance: {} ==> {}'.format(significance, decision))
    print('========================')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--systemA', type=str, help='predictions for system A')
    parser.add_argument('--systemB', type=str, help='predictions for system B')
    parser.add_argument('--truth', type=str, help='true values')
    parser.add_argument('--R', type=int, default=10000)
    args = parser.parse_args()

    a_ = load_data(args)
    a_2 = align(a_)

    global map_
    cnt = 1
    map_ = {'1:NR:2': 0}
    for l in a_2['true'].values():
        if l not in map_:
            map_[l] = cnt
            cnt += 1

    map_2 = map_.copy()

    for l in map_2.keys():
        t = l.split(':')
        if t[2]+':'+t[1]+':'+t[0] not in map_:
            map_[t[2]+':'+t[1]+':'+t[0]] = cnt
            cnt += 1

    for l1, l2, l3 in zip(a_2['A'], a_2['B'], a_2['true']):
        a_2['A'][l1] = map_[a_2['A'][l1]]
        a_2['B'][l1] = map_[a_2['B'][l1]]
        a_2['true'][l1] = map_[a_2['true'][l1]]

    print(args.systemA)
    print(args.systemB)

    print('=== OVERALL ===')
    p1, r1, f1 = eval_(a_2['A'], a_2['true']).values()
    p2, r2, f2 = eval_(a_2['B'], a_2['true']).values()
    print('System A: P = {:.2f}\tR = {:.2f}\tF1 = {:.2f}'.format(p1*100, r1*100, f1*100))
    print('System B: P = {:.2f}\tR = {:.2f}\tF1 = {:.2f}'.format(p2*100, r2*100, f2*100))
    sig_test(args, a_2['A'], a_2['B'], a_2['true'])


if __name__ == "__main__":
    main()
