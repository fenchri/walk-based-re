#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 17/06/2019

author: fenia
"""

from collections import OrderedDict
from recordtype import recordtype


EntityInfo = recordtype('EntityInfo', 'id type name start end')
PairInfo = recordtype('PairInfo', 'id type arg1 arg2 direction cross')


def read_relation_input(input_file, documents, entities, relations):
    """
    Read input file in special format
    """
    def chunks(l, n):
        """ Successive n-sized chunks from l. """
        res = []
        for i in range(0, len(l), n):
            assert len(l[i:i + n]) == n, 'sequence of invalid length'
            res += [l[i:i + n]]
        return res

    lengths = []
    with open(input_file, 'r') as infile:
        for line in infile:
            line = line.rstrip().split('\t')
            docid = line[0]
            text = line[1].split(' ')
            pairs = chunks(line[2:], 13)

            if docid not in documents:
                documents[docid] = text

            if docid not in entities:
                entities[docid] = OrderedDict()

            if docid not in relations:
                relations[docid] = OrderedDict()

            # max sentence length
            lengths += [len(text)]

            all_pairs = 0
            for p in pairs:
                if 'R' + str(all_pairs) not in relations[docid]:
                    relations[docid]['R' + str(all_pairs)] = PairInfo('R' + str(all_pairs),
                                                                      p[0], p[3], p[8], p[1], p[2])
                    all_pairs += 1

                # entities
                if p[3] not in entities[docid]:
                    entities[docid][p[3]] = EntityInfo(p[3], p[5], p[4], p[6], p[7])

                if p[8] not in entities[docid]:
                    entities[docid][p[8]] = EntityInfo(p[8], p[10], p[9], p[11], p[12])

            assert len(relations[docid]) == all_pairs, 'Not all pairs assigned'
    return lengths, documents, entities, relations

