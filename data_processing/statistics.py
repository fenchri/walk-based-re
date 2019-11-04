#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 09/05/2019

author: fenia
"""

import argparse
import numpy as np
from collections import OrderedDict
from recordtype import recordtype
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)
parser.add_argument('--out', type=str)
args = parser.parse_args()


EntityInfo = recordtype('EntityInfo', 'type mstart mend sentNo')
PairInfo = recordtype('PairInfo', 'type direction cross')


def chunks(l, n):
    """
    Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        assert len(l[i:i + n]) == n
        yield l[i:i + n]


documents = {}
entities = {}
relations = {}
count = {}
with open(args.data, 'r') as infile:

    for line in tqdm(infile):
        line = line.rstrip().split('\t')
        pairs = chunks(line[2:], 13)

        id_ = line[0]

        if id_ not in documents:
            documents[id_] = []

        for sent in line[1].split('|'):
            documents[id_] += [sent]

        if id_ not in entities:
            entities[id_] = OrderedDict()

        if id_ not in relations:
            relations[id_] = OrderedDict()

        for p in pairs:
            # pairs
            if (p[3], p[8]) not in relations[id_]:
                relations[id_][(p[3], p[8])] = PairInfo(p[0], p[1], p[2])
            else:
                print('duplicates!')

            # entities
            if p[3] not in entities[id_]:
                entities[id_][p[3]] = EntityInfo(p[5], p[6], p[7], '0')

            if p[8] not in entities[id_]:
                entities[id_][p[8]] = EntityInfo(p[10], p[11], p[12], '0')


#pos_pairs_sent = {}
#for id_ in relations.keys():
#   if 6 <= len(entities[id_]) <= 9:
#        temp = 0
#        all_ = 0
#        for k, p in relations[id_].items():
#            if p.type != '1:NR:2' and p.type != 'not_include':
#                temp += 1
#            else:
#                all_ += 1
#        if id_ not in pos_pairs_sent:
#            pos_pairs_sent[id_] = temp / (all_ + temp)

##    if id_ not in pos_pairs_sent:
##        pos_pairs_sent[id_] = 0
##    for k, p in relations[id_].items():
##        if p.type.split(':')[1] == 'ORG-AFF':
##            all_ += 1
##            if (int(entities[id_][k[0]].mend)-1 < int(entities[id_][k[1]].mstart)) and (int(entities[id_][k[1]].mstart) - int(entities[id_][k[0]].mend) < 3):
##                temp +=  1
##            elif (int(entities[id_][k[1]].mend)-1 < int(entities[id_][k[0]].mstart)) and (int(entities[id_][k[0]].mstart) - int(entities[id_][k[1]].mend) < 3):
##                temp +=  1
##          #  pos_pairs_sent[id_] += 1

##print(temp, '/', len(relations), '-->', temp/len(relations) * 100)
#print(np.mean([v for k,v in pos_pairs_sent.items()]), len(pos_pairs_sent))
#exit(0)

# write gold data
#with open(args.out, 'w') as outfile:
#    for id_ in relations.keys():
#        for k, p in relations[id_].items():
#            outfile.write('{}|{}|{}|{}\n'.format(id_, k[0], k[1], p.type))

#            if p.type.split(':')[1] != 'NR':
#                if p.type.split(':')[0] == '2' and (int(entities[id_][k[0]].mend)-1 < int(entities[id_][k[1]].mstart)):
#                    if p.type.split(':')[1] not in count:
#                        count[p.type.split(':')[1]] = 1
#                    else:
#                        count[p.type.split(':')[1]] += 1

#            if p.direction == 'L2R':
#                if p.type == 'not_include':
#                    continue
#                elif p.type == '1:NR:2':
#                    outfile.write('{}|{}|{}|{}\n'.format(id_, k[0], k[1], p.type))
#                else:
#                    outfile.write('{}|{}|{}|{}\n'.format(id_, k[0], k[1], p.type.split(':')[1]))
#            else:
#                if p.type == 'not_include':
#                    continue
#                elif p.type == '1:NR:2':
#                    outfile.write('{}|{}|{}|{}\n'.format(id_, k[1], k[0], p.type))
#                else:
#                    outfile.write('{}|{}|{}|{}\n'.format(id_, k[1], k[0], p.type.split(':')[1]))

      
docs = len(documents)
pair_types = {}
ent_types = {}
men_types = {}
for id_ in relations.keys():
    for k, p in relations[id_].items():
        if p.type == '1:NR:2':
            p_ = p.type
        elif p.type == 'not_include':
            p_ = p.type
        elif len(p.type.split(':')) == 3:
            p_ = p.type.split(':')[1]
        else:
            p_ = ':'.join(p.type.split(':')[1:-1])

        if p_ not in pair_types:
            pair_types[p_] = 0
        pair_types[p_] += 1

    for e in entities[id_].values():
        if e.type not in ent_types:
            ent_types[e.type] = 0
        ent_types[e.type] += 1


ents_per_doc = [len(entities[n]) for n in documents.keys()]
sents_per_doc = [len(s[0].split(' ')) for s in documents.values()]

print('Sentences\t{}'.format(len(documents)))
print('Pairs\t{}'.format(sum([y for x, y in pair_types.items() if x != '1:NR:2'])))
for x in ['\t{:<10}\t{}'.format(k, v) for k, v in sorted(pair_types.items())]:
    print(x)
print('% Negative pairs\t{:.2f}'.format(pair_types['1:NR:2']*100/sum([y for x, y in pair_types.items()])))
print()

print('Entities\t{}'.format(sum([y for x, y in ent_types.items()])))
for x in ['\t{:<10}\t{}'.format(k, v) for k, v in sorted(ent_types.items())]:
    print(x)
print()

print('Average sentence length\t{:.2f}'.format(np.mean(sents_per_doc)))
print('Average entities/sentence\t{:.2f}'.format(np.mean(ents_per_doc)))

