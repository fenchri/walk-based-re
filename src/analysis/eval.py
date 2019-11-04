#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 24/09/2019

author: fenia
"""

import sys
import numpy as np
from tqdm import tqdm


dic = {'P17': 'Country',
       'P131': 'Located in',
       'P47': 'Shares border',
       'P31': 'Instance of',
       'P641': 'Sport',
       'P27': 'Citizenship',
       'P361': 'Part of',
       'P279': 'Subclass of'}


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
        for line in tqdm(preds_file):
            line = line.rstrip().split('|')

            doc_id = line[0]  # .split('.split')[0]

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
        for line2 in tqdm(gold_file):
            line2 = line2.rstrip().split('|')

            doc_id = line2[0]  #.split('**')[0]

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

    for l in tqdm(labels):
        if l != '1:NR:2':
            tp[l] += len([a for a in pr if a in gl and a[3] == l])
            fp[l] += len([a for a in pr if a not in gl and a[3] == l])
            fn[l] += len([a for a in gl if a not in pr and a[3] == l])
            classes[l] = prf(tp[l], fp[l], fn[l])

    tp_tot = np.sum([tp[a] for a in tp])
    fp_tot = np.sum([fp[a] for a in fp])  # len([a for a in pr if a not in gl])
    fn_tot = np.sum([fn[a] for a in fn])  # len([a for a in gl if a not in pr])

    total = prf(tp_tot, fp_tot, fn_tot)
    return total, classes


total, classes = evaluation(sys.argv[1], sys.argv[2])

print('{:<10} \t P\tR\tF1'.format(' '))
avg = []
for c in classes:
    if c == 'P17':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P131':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P47':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P31':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P641':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P27':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P361':
        print(
            '{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]
    elif c == 'P279':
        print('{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(dic[c], classes[c][0]*100, classes[c][1]*100, classes[c][2]*100))
        avg += [[classes[c][0], classes[c][1], classes[c][2]]]

    # print('{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format(c, classes[c][0] * 100, classes[c][1] * 100, classes[c][2] * 100))
    # avg += [[classes[c][0], classes[c][1], classes[c][2]]]

avg = np.array(avg)
avg = np.mean(avg, axis=0)

print('{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format('Micro', total[0]*100, total[1]*100, f1(total[0], total[1])*100))
print('{:<10} \t {:.2f}\t{:.2f}\t{:.2f}'.format('Macro', avg[0]*100, avg[1]*100, f1(avg[0], avg[1])*100))
