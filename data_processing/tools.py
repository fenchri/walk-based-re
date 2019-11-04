#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import os
import re
import sys
from recordtype import recordtype
from networkx.algorithms.components.connected import connected_components
from itertools import combinations, permutations
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
from utils import using_split2


TextStruct = recordtype('TextStruct', 'docid txt')
EntStruct = recordtype('EntStruct', 'docid entid name off1 off2 type sent_no word_id')
RelStruct = recordtype('RelStruct', 'docid relid type arg1 arg2')
PairStruct = recordtype('PairStruct', 'docid type arg1 arg2 dir cross')


def generate_pairs(uents, true_rels):
    """
    Generate pais based on their direction as well L2R or R2L
    """
    pairs = OrderedDict()
    combs = combinations(uents, 2)

    count_not_found = 0
    unk = 0
    total_rels = len(true_rels)
    found_rels = 0

    for c in combs:
        target = [c[0], c[1]]

        if target[0].word_id[-1] <= target[1].word_id[0]:  # A before B (in text)
            a1 = target[0]
            a2 = target[1]
        else:                                              # B before A (in text)
            a1 = target[1]
            a2 = target[0]

        not_found_rels = 0

        for tr in true_rels:
            # AB existing relation
            if (tr.arg1 == a1.entid) and (tr.arg2 == a2.entid):
                if tr.type == 'Other':
                    pairs[tr.relid] = PairStruct(tr.docid, '1:NR:2', a1, a2, 'L2R', 'NON-CROSS')
                    found_rels += 1
                else:
                    pairs[tr.relid] = PairStruct(tr.docid, '1:'+tr.type+':2', a1, a2, 'L2R', 'NON-CROSS')
                    found_rels += 1

            # BA existing relation 
            elif (tr.arg1 == a2.entid) and (tr.arg2 == a1.entid):
                if tr.type == 'Other':
                    pairs[tr.relid] = PairStruct(tr.docid, '1:NR:2', a1, a2, 'R2L', 'NON-CROSS')
                    found_rels += 1
                else:
                    pairs[tr.relid] = PairStruct(tr.docid, '2:'+tr.type+':1', a1, a2, 'R2L', 'NON-CROSS')
                    found_rels += 1

            # relation not found
            else:
                not_found_rels += 1

        # this pair does not have a relation
        if not_found_rels == total_rels:
            # pairs['R_neg'+str(unk)] = PairStruct(a1.docid, '1:NR:2', a1, a2, 'L2R', 'NON-CROSS')
            pairs['R_neg'+str(unk)] = PairStruct(a1.docid, 'not_include', a1, a2, 'L2R', 'NON-CROSS')
            unk += 1
            #     elif c[1].type == t1 and c[0].type == t2:
            #       pairs[(c[1], c[0])] = PairStruct(a1.pmid, '1:NR:2', c[1], c[0], 'R2L', cross_res, (a2, a1))
            #       unk += 1

    if found_rels != total_rels:
        count_not_found += (total_rels-found_rels)
#        tqdm.write('FOUND {} <> TOTAL {}, diff {}'.format(found_rels, total_rels, total_rels-found_rels))
#        for p in true_rels:
#            if p.relid not in pairs:
#                tqdm.write('{}, {}'.format(p.arg1, p.arg2))
    return pairs, count_not_found


def generate_pairs_types(uents, type1, type2, true_rels):
    """
    Generate pais based on their direction as well L2R or R2L
    """
    pairs = OrderedDict()
    combs = combinations(uents, 2)

    count_not_found = 0
    unk = 0
    total_rels = len(true_rels)
    found_rels = 0

    for c in combs:
        a1 = c[0]
        a2 = c[1]
        
        not_found_rels = 0
        for tr in true_rels:
            if (tr.arg1 == a1.entid) and (tr.arg2 == a2.entid):
                pairs[tr.relid] = PairStruct(tr.docid, '1:'+tr.type+':2', a1, a2, 'L2R', 'NON-CROSS')
                found_rels += 1

            elif (tr.arg2 == a1.entid) and (tr.arg1 == a2.entid):
                pairs[tr.relid] = PairStruct(tr.docid, '2:'+tr.type+':1', a1, a2, 'R2L', 'NON-CROSS')
                found_rels += 1

            # relation not found
            else:
                not_found_rels += 1

        # this pair does not have a relation
        if not_found_rels == total_rels:
            for t1, t2 in zip(type1, type2):
                if (a1.type == t1) and (a2.type == t2):
                    pairs['R_neg'+str(unk)] = PairStruct(a1.docid, '1:NR:2', a1, a2, 'L2R', 'NON-CROSS')
                    unk += 1
       
    if found_rels != total_rels:
        count_not_found += (total_rels-found_rels)
        tqdm.write('FOUND {} <> TOTAL {}, diff {}'.format(found_rels, total_rels, total_rels-found_rels))
        for p in true_rels:
            if p.relid not in pairs:
                tqdm.write('{}, {}'.format(p.arg1, p.arg2))
    return pairs, count_not_found


def fix_sent_break(sents, entities):
    """
    Fix sentence break if inside an entity.
    Args:
        sents: (list) old sentences
        entities: (list of structs) entities

    Returns: (list) new sentences
    """
    sents_break = '\n'.join(sents)
    for e in entities:
        if '\n' in sents_break[e.off1:e.off2]:
            sents_break = sents_break[0:e.off1] + sents_break[e.off1:e.off2].replace('\n', '') + sents_break[e.off2:]
    return sents_break.split('\n')


def adjust_offsets(old_sents, new_sents, old_entities):
    """
    Adjust offsets  of entities
    Args:
        old sents: (list) old, non-tokenized sentences
        new_sents: (list) new, tokenized sentences
        old_entities: (list of structs) entities with old offsets

    Returns: (list of struces) entities with new offsets
    """
    original = " ".join(old_sents)
    newtext = " ".join(new_sents)
    new_entities = []
    terms = {}
    for e in old_entities:
        start = int(e.off1)
        end = int(e.off2)

        if (start, end) not in terms:
            terms[(start, end)] = [[start, end, e.type, e.name, e.docid, e.entid]]
        else:
            terms[(start, end)].append([start, end, e.type, e.name, e.docid, e.entid])

    orgidx = 0
    newidx = 0
    orglen = len(original)
    newlen = len(newtext)

    terms2 = terms.copy()
    while orgidx < orglen and newidx < newlen:
        # print(repr(original[orgidx]), orgidx, repr(newtext[newidx]), newidx)

        if original[orgidx] == newtext[newidx]:
            orgidx += 1
            newidx += 1
        elif newtext[newidx] == '\n':
            newidx += 1
        elif original[orgidx] == '\n':
            orgidx += 1
        elif original[orgidx] == ' ':
            orgidx += 1
        elif newtext[newidx] == ' ':
            newidx += 1
        else:
            tqdm.write("Non-existent text: %d\t --> %s != %s " % (orgidx, repr(original[orgidx-10:orgidx+10]),
                                                                          repr(newtext[newidx-10:newidx+10])))
            exit(0)

        starts = [key[0] for key in terms2.keys()]
        ends = [key[1] for key in terms2.keys()]

        if orgidx in starts:
            tt = [key for key in terms2.keys() if key[0] == orgidx]  # take all pairs with start == orgidx
            for sel in tt:
                for l in terms[sel]:
                    l[0] = newidx
                for l in terms2[sel]:
                    l[0] = newidx

        if orgidx in ends:
            tt2 = [key for key in terms2.keys() if key[1] == orgidx]  # take all pairs with end == orgidx
            for sel2 in tt2:
                for l in terms[sel2]:
                    l[1] = newidx
                for l in terms2[sel2]:
                    if l[1] == orgidx:
                        l[1] = newidx

            for t_ in tt2:
                del terms2[t_]

    for ts in terms.values():
        for term in ts:
            if newtext[term[0]:term[1]].replace(" ", "").replace("\n", "") == \
                    term[3].replace(" ", "").replace('\n', ""):

                new_entities += [EntStruct(term[4], term[5], newtext[term[0]:term[1]], term[0], term[1], term[2], -1, [])]
            else:
                tqdm.write('ERROR: {} ({}-{}) <=> {}'.format(repr(newtext[term[0]:term[1]]), term[0], term[1],
                                                        repr(term[3])))
    return new_entities


def offsets2tokids(sents, entities):
    """
    Convert entities to token Ids
    Args:
        sents:
        entities:

    Returns:
    """
    text = " ".join(sents)
    
    for e in entities:
        
        span2append = []
        for tok_id, (tok, start, end) in enumerate(using_split2(text)):
            start = int(start)
            end = int(end)

            if (start, end) == (e.off1, e.off2):
                span2append.append(tok_id)

            elif start == e.off1 and end < e.off2:
                span2append.append(tok_id)

            elif start > e.off1 and end < e.off2:
                span2append.append(tok_id)

            elif start > e.off1 and end == e.off2:
                span2append.append(tok_id)

            elif len(set(range(start, end)).intersection(set(range(e.off1, e.off2)))) > 0:
                span2append.append(tok_id)

                # entity has more characters (incomplete tokenization)
                tqdm.write('entity: {:<10} ({}-{}) <-> token: {:<10} ({}-{}) <-> final: {:<10}'.format(
                    text[e.off1:e.off2], e.off1, e.off2, tok, start, end,
                    ' '.join(text.split(' ')[span2append[0]:span2append[-1] + 1])))

        # include all tokens!
        if len(span2append) != len(text[e.off1:e.off2].split(' ')):
            tqdm.write('DOC_ID {}, new entity {}, tokens {}, old entity {}'.format(
                e.docid, repr(text[e.off1:e.off2]), span2append, repr(e.name)))
            tqdm.write(using_split2(text))
        else:
            e.word_id = span2append
    return entities


def ent2sent(sents, entities):
    """
    Find the sentence where the entity belongs.
    Args:
        sents:
        entities:

    Returns:
    """
    cur = 0
    new_sent_range = []
    for s in sents:
        new_sent_range += [(cur, cur + len(s))]
        cur += len(s) + 1

    for e in entities:
        sent_no = []
        for s_no, sr in enumerate(new_sent_range):
            if set(np.arange(e.off1, e.off2)).issubset(set(np.arange(sr[0], sr[1]))):
                sent_no += [s_no]

        if len(sent_no) == 1:
            e.sent_no = sent_no[0]
        else:
            tqdm.write('{} ({}, {}) -- {}'.format(sent_no, e.off1, e.off2, new_sent_range))
    return entities


def check_entities(entities, relations, include_nested=True):
    """
    Remove -- duplicate entries
           -- nested entities
           -- entities that are not found in text
    Args:
        entities:  (list of structs) entities
        include_nested: (bool) include/not nested entities

    Returns: (list of structs) entities

    """
    todel = []
    if not entities:
        pass
    else:
        for a, b in combinations(entities, 2):
            overlap = set(a.word_id) & set(b.word_id)
            if set(a.word_id) == set(b.word_id):  #  same ids exactly
                in_r = False
                for r in relations:
                    if r.arg1 == a.entid:
                        e2r = b
                        in_r = True
                        other = a
                    elif r.arg1 == b.entid:
                        e2r = a
                        in_r = True
                        other = b
                    elif r.arg2 == a.entid:
                        e2r = b
                        in_r = True
                        other = a
                    elif r.arg2 == b.entid:
                        e2r = a
                        in_r = True
                        other = b
                if in_r == False:
                    e2r = b
                    other = a
                todel += [('Entity {} in doc {} is same with {} -> ignore'.format(repr(e2r.name), e2r.docid, repr(other.name)), e2r)]
        
#            elif bool(overlap):  # overlapping ranges

#                # partial overlap -- ignore anyway
#                if (len(list(overlap)) != len(a.word_id)) and (len(list(overlap)) != len(b.word_id)):
#                    if len(a.word_id) > len(b.word_id):
#                        other = a
#                        e2r = b
#                    else:
#                        e2r = a  # remove shorter entity
#                        other = b
#                    if e2r not in todel:
#                        todel += [('Entity {} in doc {} partially overlaps with {} -> ignore'.format(repr(e2r.name), e2r.docid, repr(other.name)), e2r)]

            # nested
            if (len(list(overlap)) == len(a.word_id)) or (len(list(overlap)) == len(b.word_id)):
                if include_nested:
                    continue
                else:
                    in_r = False
                    for r in relations:
                        if r.arg1 == a.entid:
                            e2r = b
                            in_r = True
                        elif r.arg1 == b.entid:
                            e2r = a
                            in_r = True
                        elif r.arg2 == a.entid:
                            e2r = b
                            in_r = True
                        elif r.arg2 == b.entid:
                            e2r = a
                            in_r = True
                    if in_r == False:
                        if len(a.word_id) > len(b.word_id):
                            e2r = b
                        else:
                            e2r = a  # remove shorter entity
                    if e2r not in todel:
                        todel += [('Entity {} in doc {} is nested -> ignore'.format(e2r.entid, e2r.docid), e2r)]

        for e in entities:
            if not e.word_id:
                if e not in todel:
                    todel += [('Entity {} in doc {} not found in text -> ignore'.format(e.entid, e.docid), e)]

    for txt, td in todel:
        tqdm.write(txt)
        if td in entities:
            entities.remove(td)
    return entities, len(todel)


def check_relations(entities, relations):
    """
    Remove -- duplicate entries
           -- relations with missing arguments
    Args:
        entities: (list of structs) entities
        relations: (list of structs) relations

    Returns: (list of structs) relations

    """
    todel = []
    # check if entities are missing
    for r in relations:
        okA = False
        okB = False
        for e in entities:
            if r.arg1 == e.entid:
                okA = True

            if r.arg2 == e.entid:
                okB = True

            if okA and okB:
                break

        if okA and okB:
            pass
        else:
            todel += [('Relation {} in doc {} misses an entity -> ignore <<----------'.format(r.relid, r.docid), r)]

    # check for duplicates
    for a, b in combinations(relations, 2):
        if (a.type == b.type) and (a.arg1 == b.arg1) and (a.arg2 == b.arg2):
            todel += [('Relation {} in doc {} is duplicate -> ignore <<----------'.format(a.relid, r.docid), a)]
        elif (a.type != b.type) and (a.arg1 == b.arg1) and (a.arg2 == b.arg2):
            todel += [('Relation {} in doc {} is duplicate -> ignore <<----------'.format(a.relid, r.docid), a)]
        elif (a.type != b.type) and (a.arg1 == b.arg2) and (a.arg2 == b.arg1):
            todel += [('Relation {} in doc {} is duplicate -> ignore <<----------'.format(a.relid, r.docid), a)]
        elif (a.type == b.type) and (a.arg1 == b.arg2) and (a.arg2 == b.arg1):
            todel += [('Relation {} in doc {} is duplicate -> ignore <<----------'.format(a.relid, r.docid), a)]

    for txt, tdr in todel:
        tqdm.write(txt)
        relations.remove(tdr)
    return relations, len(todel)


def doc2sent(entities, relations, sentences):
    """
    Map entities and relations to sentences.
    Args:
        entities: (list of structs) entities in document
        relations: (list of structs) relations in document
        sentences: (list) sentences in documents

    Returns: (list of structs) entities in sents,
             (list of structs) relations in sents,
             (list) sents
    """
    new_entities = OrderedDict()
    new_relations = OrderedDict()
    new_sents = OrderedDict()

    total_r = 0
    for i, s in enumerate(sentences):
        new_sents[i] = s
        new_entities[i] = []
        new_relations[i] = []

        included_ents = []
        for e in entities:
            if e.sent_no == i:
                take = e
                # TODO change the offsets as well
                take.word_id = [int(w - np.sum([len(s.split(' ')) for s in sentences[:i]])) for w in e.word_id]
                new_entities[i] += [take]
                included_ents += [take.entid]

        for r in relations:
            if (r.arg1 in included_ents) and (r.arg2 in included_ents):
                new_relations[i] += [r]

        total_r += len(new_relations[i]) 
    
    return new_sents, new_entities, new_relations

