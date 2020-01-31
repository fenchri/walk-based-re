#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 03/06/2019
author: fenia
"""

from tqdm import tqdm
import argparse
from glob import glob
from recordtype import recordtype
import os
from collections import OrderedDict
from utils import sentence_split_genia, sentence_split_stanford, tokenize_genia, tokenize_stanford
from tools import fix_sent_break, adjust_offsets, check_entities, check_relations, generate_pairs, generate_pairs_types
from tools import ent2sent, doc2sent, offsets2tokids


TextStruct = recordtype('TextStruct', 'docid txt')
EntStruct = recordtype('EntStruct', 'docid entid name off1 off2 type sent_no word_id')
RelStruct = recordtype('RelStruct', 'docid relid type arg1 arg2')


def write2file(new_sents, new_entities, new_relations, args, doc_id, data_out, not_found, positive, negative, no_pairs):
    # generate pairs
    new_entities, missed_ents = check_entities(new_entities, new_relations, include_nested=args.nested)
    new_relations, missed_rels = check_relations(new_entities, new_relations)

    if args.type1 and args.type2:
        pairs, cnf = generate_pairs_types(new_entities, args.type1, args.type2, new_relations)
    else:
        pairs, cnf = generate_pairs(new_entities, new_relations)
    not_found += cnf

    if not pairs:
        no_pairs += 1
    else:
        data_out.write('{}\t{}'.format(doc_id, new_sents.lower()))

        for args_, p in pairs.items():
            if p.type != '1:NR:2' and p.type != 'not_include':
                positive += 1
            elif p.type == '1:NR:2':
                negative += 1

            data_out.write('\t{}\t{}\t{}'.format(p.type, p.dir, p.cross))
            data_out.write('\t{}\t{}\t{}\t{}\t{}'.format(p.arg1.entid, p.arg1.name.lower(), p.arg1.type,
                                                         p.arg1.word_id[0], p.arg1.word_id[-1]+1))
            data_out.write('\t{}\t{}\t{}\t{}\t{}'.format(p.arg2.entid, p.arg2.name.lower(), p.arg2.type,
                                                         p.arg2.word_id[0], p.arg2.word_id[-1]+1))
        data_out.write('\n')
    return not_found, positive, negative, no_pairs


def read_brat(args):
    if not os.path.exists(args.output_file+'_files'):
        os.makedirs(args.output_file+'_files')

    abstracts = OrderedDict()
    entities = OrderedDict()
    relations = OrderedDict()

    for filef in tqdm(glob(args.input_folder+'*.txt'), desc='Reading'):
        filename = filef.split('/')[-1].split('.txt')[0]

        if filename not in abstracts:
            abstracts[filename] = []
        if filename not in entities:
            entities[filename] = []
        if filename not in relations:
            relations[filename] = []

        with open(filef, 'r') as infile:
            for line in infile:
                abstracts[filename] += [TextStruct(filename, line.rstrip())]

        with open(args.input_folder+filename+'.ann', 'r') as infile:
            for line in infile:
                if line.startswith('T'):
                    line = line.rstrip().split('\t', 2)
                    region = line[1].split(' ')
                    entities[filename] += [EntStruct(filename, line[0], line[2], int(region[1]), int(region[2]),
                                                     region[0], -1, [])]

                elif line.startswith('R'):
                    line = line.rstrip().split('\t')
                    region = line[1].split(' ')
                    relations[filename] += [RelStruct(filename, line[0], region[0], region[1].split('Arg1:')[1],
                                                      region[2].split('Arg2:')[1])]

    if args.level == 'doc':
        print('# Documents: {}'.format(len(abstracts)))
    elif args.level == 'sent':
        print('# Sentences: {}'.format(len(abstracts)))

    print('# Entities: {}'.format(sum([len(e) for e in entities.values()])))
    print('# Pairs: {}'.format(sum([len(r) for r in relations.values()])))
        
    return abstracts, entities, relations


def main():
    """
    Main processing function
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', '-i', type=str)
    parser.add_argument('--output_file', '-o', type=str)
    parser.add_argument('--level', choices=['doc', 'sent'])
    parser.add_argument('--domain', choices=['gen', 'bio'])
    parser.add_argument('--processed', action='store_true')
    parser.add_argument('--nested', action='store_true')
    parser.add_argument('--type1', nargs='*')
    parser.add_argument('--type2', nargs='*')
    args = parser.parse_args()

    abstracts, entities, relations = read_brat(args)
    not_found, positive, negative, no_pairs = 0, 0, 0, 0
    all_ents = 0

    pbar = tqdm(list(abstracts.keys()))
    with open(args.output_file + '.data', 'w') as data_out:
        for i in pbar:
            pbar.set_description("Processing DOC_ID {}".format(i))

            if not args.processed:
                # sentence splitting
                orig_sentences = [item for sublist in [a.txt.split('\n') for a in abstracts[i]] for item in sublist]
                orig_sentences = [item.replace('', ' ') if item == '' else item for item in orig_sentences]

                if args.domain == 'bio':
                    split_sents = sentence_split_genia(orig_sentences)
                else:
                    split_sents = sentence_split_stanford(orig_sentences)
                split_sents = fix_sent_break(split_sents, entities[i])

                with open(args.output_file + '_files/' + i + '.split.txt', 'w') as f:
                    f.write('\n'.join(split_sents))

                # tokenisation
                if args.domain == 'bio':
                    token_sents = tokenize_genia(split_sents)
                else:
                    token_sents = tokenize_stanford(split_sents)

                with open(args.output_file + '_files/' + i + '.split.tok.txt', 'w') as f:
                    f.write('\n'.join(token_sents))

                # adjust offsets
                new_entities = adjust_offsets(orig_sentences, token_sents, entities[i])
                new_entities = offsets2tokids(token_sents, new_entities)
                new_entities = ent2sent(token_sents, new_entities)
                sentences = token_sents

            else:
                new_entities = entities[i]
                sentences = [item for sublist in [a.txt.split('\n') for a in abstracts[i]] for item in sublist]
                new_entities = offsets2tokids(sentences, new_entities)
                new_entities = ent2sent(sentences, new_entities)

            # from doc to sentence
            if args.level == 'doc':
                new_sents, new_entities, new_relations = doc2sent(new_entities, relations[i], sentences)

                for s_id in new_sents.keys():
                    not_found, positive, negative, no_pairs = \
                        write2file(new_sents[s_id], new_entities[s_id], new_relations[s_id],
                                   args, str(i)+'**'+str(s_id),
                                   data_out,
                                   not_found, positive, negative, no_pairs)

            else:
                new_relations = relations[i]
                not_found, positive, negative, no_pairs = \
                    write2file(' '.join(sentences), new_entities, new_relations, args, i,
                               data_out, not_found, positive, negative, no_pairs)
            all_ents += len(new_entities)

    print('Total positive pairs:', positive)
    print('Total negative pairs:', negative)
    print('Total not found pairs:', not_found)
    print('Sentences without pairs:', no_pairs)
    print('Entities:', all_ents)


if __name__ == "__main__":
    main()
