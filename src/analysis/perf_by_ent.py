#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 18/09/2019

author: fenia
"""

import sys
import os
import numpy as np
import argparse
from collections import OrderedDict
sys.path.append('..')
import reader
from tqdm import tqdm

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


number_of_entities = OrderedDict()

print('Total sentences: {}'.format(len(sentences)))

#for i in range(2, 50):
number_of_entities[2] = []
number_of_entities[3] = []
number_of_entities[4] = []
number_of_entities[6] = []
number_of_entities[10] = []

#for i in range(2, 50):
for key in sentences.keys():
    ent_num = len(entities[key])

    # if ent_num == i:
    #     number_of_entities[i] += [key]

    # if ent_num < 100:
    #     number_of_entities[2] += [key]

    if ent_num == 2:
        number_of_entities[2] += [key]

    elif ent_num == 3:
        number_of_entities[3] += [key]

    elif 4 <= ent_num <= 5:
        number_of_entities[4] += [key]

    elif 6 <= ent_num <= 9:
        number_of_entities[6] += [key]

   # elif 9 <= ent_num <= 11:
   #     number_of_entities[9] += [key]

    elif ent_num >= 10:
        number_of_entities[10] += [key]


# select sentences and test
print('{:<5}\t{:<5}\t{:<5}\t{:<5}\t{:<5}'.format('# Ents', '# Sents', 'P', 'R', 'F1'))
print('{:<5}\t{:<5}\t{:<5}\t{:<5}\t{:<5}'.format('-----', '-----', '-----', '-----', '-----'))

for en in number_of_entities.keys():
    with open('temp.gold', 'w') as outfile, open(args.truth, 'r') as infile:
        gold = [line for line in infile]
        gold_names = [g.rstrip().split('|')[0] for g in gold]

        tmp = frozenset(number_of_entities[en])
        for j, m in enumerate(gold_names):
            if m in tmp:
                outfile.write(gold[j])

    with open('temp.pred', 'w') as outfile, open(args.preds, 'r') as infile:
        preds = [line for line in infile]
        preds_names = [p.rstrip().split('|')[0] for p in preds]
        
        for j, m in enumerate(preds_names):
            if m in tmp:
                outfile.write(preds[j])

    a, b, micro, macro = evaluation('temp.pred', 'temp.gold')

    print('{:<5}\t{:<5}\t{:.2f}\t{:.2f}\t{:.2f}'.format(en, len(number_of_entities[en]),
                                                        micro[0]*100, micro[1]*100, micro[2]*100))







