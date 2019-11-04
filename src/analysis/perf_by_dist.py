#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/09/2019

author: fenia
"""

import sys
import os
import numpy as np
import argparse
from collections import OrderedDict
sys.path.append('..')
import reader

"""
Performance as a function of the number of entities per sentence.
"""

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, help='original data')
parser.add_argument('--preds', type=str, help='predictions file')
parser.add_argument('--truth', type=str, help='truth file')
args = parser.parse_args()

lengths, sentences, entities, pairs = reader.read_relation_input(args.data,
                                                                 OrderedDict(), OrderedDict(), OrderedDict())


def f1(p, r):
    return ((2 * p * r) / (p + r)) if p != 0.0 and r != 0.0 else 0.0


def prf(tp, fp, fn):
    p = float(tp) / (tp + fp) if (tp + fp != 0) else 0.0
    r = float(tp) / (tp + fn) if (tp + fn != 0) else 0.0
    f = ((2 * p * r) / (p + r)) if p != 0.0 and r != 0.0 else 0.0
    return [p, r, f]


def evaluation(preds, gold):
    pr, gl, labels = [], [], []
    with open(preds, 'r') as preds_file:
        for line in preds_file:
            line = line.rstrip().split('|')
            doc_id = line[0]

            if line[3] == '1:NR:2':
                arg1 = line[1]
                arg2 = line[2]
                label = line[3]
            elif ':' not in line[3]:
                arg1 = line[1]
                arg2 = line[2]
                label = line[3]
            else:
                l = line[3].split(':')
                if l[1] != 'NR' and l[0] == '1':
                    arg1 = line[1]
                    arg2 = line[2]
                    label = l[1]
                elif l[1] != 'NR' and l[0] == '2':
                    arg1 = line[2]
                    arg2 = line[1]
                    label = l[1]
                else:
                    continue

            pr += [(doc_id, arg1, arg2, label)]

    with open(gold, 'r') as gold_file:
        for line2 in gold_file:
            line2 = line2.rstrip().split('|')

            doc_id = line2[0]

            if line2[3] == '1:NR:2':
                arg1 = line2[1]
                arg2 = line2[2]
                label = line2[3]
            elif ':' not in line2[3]:
                arg1 = line2[1]
                arg2 = line2[2]
                label = line2[3]
            else:
                l2 = line2[3].split(':')
                if l2[1] != 'NR' and l2[0] == '1':
                    arg1 = line2[1]
                    arg2 = line2[2]
                    label = l2[1]
                elif l2[1] != 'NR' and l2[0] == '2':
                    arg1 = line2[2]
                    arg2 = line2[1]
                    label = l2[1]
                else:
                    continue

            if label not in labels:
                labels += [label]
            gl += [(doc_id, arg1, arg2, label)]

    tp, fp, fn = {}, {}, {}
    classes = {}

    gl = frozenset(gl)  # in order to be faster
    pr = frozenset(pr)
    for l in labels:
        tp[l] = 0
        fp[l] = 0
        fn[l] = 0

    for l in labels:
        if l != '1:NR:2':
            tp[l] += len([a for a in pr if a in gl and a[3] == l])
            fp[l] += len([a for a in pr if a not in gl and a[3] == l])
            fn[l] += len([a for a in gl if a not in pr and a[3] == l])
            classes[l] = prf(tp[l], fp[l], fn[l])

    tp_tot = np.sum([tp[a] for a in tp])
    fp_tot = np.sum([fp[a] for a in fp])  # len([a for a in pr if a not in gl])
    fn_tot = np.sum([fn[a] for a in fn])  # len([a for a in gl if a not in pr])

    total = prf(tp_tot, fp_tot, fn_tot)
    return len(pr), len(gl), total, classes


distance = OrderedDict()
for i in range(0, 60):
    distance[i] = {}

for i in range(0, 60):
    for key in pairs.keys():
        for p in pairs[key].values():
            if p.type == 'not_include':
                continue

            if int(entities[key][p.arg1].end)-1 < int(entities[key][p.arg2].start):
                if 0 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 0:
                    distance[0].update({(key, p.arg1, p.arg2): p.type})
                elif 1 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 5:
                    distance[1].update({(key, p.arg1, p.arg2): p.type})
                elif 6 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 10:
                    distance[6].update({(key, p.arg1, p.arg2): p.type})
                elif 11 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 15:
                    distance[11].update({(key, p.arg1, p.arg2): p.type})
                elif 16 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 20:
                    distance[16].update({(key, p.arg1, p.arg2): p.type})
                elif 21 <= abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) <= 25:
                    distance[21].update({(key, p.arg1, p.arg2): p.type})
                elif abs(int(entities[key][p.arg2].start) - (int(entities[key][p.arg1].end))) >= 26:
                    distance[26].update({(key, p.arg1, p.arg2): p.type})
            else:
                if 0 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 0:
                    distance[0].update({(key, p.arg1, p.arg2): p.type})
                elif 1 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 5:
                    distance[1].update({(key, p.arg1, p.arg2): p.type})
                elif 6 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 10:
                    distance[6].update({(key, p.arg1, p.arg2): p.type})
                elif 11 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 15:
                    distance[11].update({(key, p.arg1, p.arg2): p.type})
                elif 16 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 20:
                    distance[16].update({(key, p.arg1, p.arg2): p.type})
                elif 21 <= abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) <= 25:
                    distance[21].update({(key, p.arg1, p.arg2): p.type})
                elif abs(int(entities[key][p.arg1].start) - (int(entities[key][p.arg2].end))) >= 26:
                    distance[26].update({(key, p.arg1, p.arg2): p.type})


# select sentences and test
print('{:<10}\t{:<10}\t{:<5}\t{:<5}\t{:<5}\t\t{:<5}\t{:<5}'.format('Distance', '# Pairs', 'P', 'R', 'F1', 'preds', 'gold'))
print('{:<10}\t{:<10}\t{:<5}\t{:<5}\t{:<5}'.format('-----', '-----', '-----', '-----', '-----'))

for en in [0, 1, 6, 11, 16, 21, 26]:
    tmp = frozenset(distance[en])
    pos_pairs = sum([1 for k in distance[en] if distance[en][k] != '1:NR:2' and distance[en][k] != 'not_include'])

    with open('temp.gold', 'w') as outfile:
        for k in distance[en]:
            if distance[en][k] != 'not_include':
                outfile.write('{}|{}|{}|{}\n'.format(k[0], k[1], k[2], distance[en][k]))

    with open('temp.pred', 'w') as outfile, open(args.preds, 'r') as infile:
        pred = {}
        for line in infile:
            tt = tuple(line.rstrip().split('|')[0:3])
            pred[tt] = line.rstrip().split('|')[3]

        for k in pred:
            if k in tmp:
                outfile.write('{}|{}|{}|{}\n'.format(k[0], k[1], k[2], pred[k]))
            elif (k[0], k[2], k[1]) in tmp:
                outfile.write('{}|{}|{}|{}\n'.format(k[0], k[1], k[2], pred[k]))

    len_preds, len_gold, micro, macro = evaluation('temp.pred', 'temp.gold')
    print('{:<10}\t{:<10}\t{:.2f}\t{:.2f}\t{:.2f}\t\t{:<5}\t{:<5}'.format(
        en, pos_pairs, micro[0]*100, micro[1]*100, micro[2]*100, len_preds, len_gold))

