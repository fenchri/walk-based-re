#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: fenia
"""

import os, sys, re
sys.path.append('./common/geniatagger-3.0.2/')
sys.path.append('./common/geniass/')
pwd = '/'.join(os.path.realpath(__file__).split('/')[:-1])
from geniatagger_wrapper import GENIATagger
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:9000')

properties_ss = {
  'annotators': 'ssplit',
  'ssplit.isOneSentence': 'true',
  'outputFormat': 'json'
}
# 'ssplit.newlineIsSentenceBreak=always',

properties_tok = {
  'annotators': 'tokenize',
  'outputFormat': 'json'
}

genia_splitter = os.path.join("./common/", "geniass")
exec_genia = os.path.join("./common/", "geniatagger-3.0.2", "geniatagger")
tagger = GENIATagger(exec_genia)


def using_split2(line, _len=len):
    """
    Credits to https://stackoverflow.com/users/1235039/aquavitae

    :param line: sentence
    :return: a list of words and their indexes in a string.
    """
    words = line.split(' ')
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset))
    return offsets


def replace2symbol(string):
    string = string.replace('”', '"').replace('’', "'").replace('–', '-').replace('‘', "'").replace('‑', '-').replace(
        '\x92', "'").replace('»', '"').replace('—', '-').replace('\uf8fe', ' ').replace('«', '"').replace('\uf8ff',
                                                                                                          ' ').replace(
        '£', '#').replace('\u2028', ' ').replace('\u2029', ' ')

    return string


def replace2space(string):
    spaces = ["\r", '\xa0', '\xe2\x80\x85', '\xc2\xa0', '\u2009', '\u2002', '\u200a', '\u2005', '\u2003', '\u2006', 'Ⅲ',
              '…', 'Ⅴ', "\u202f", "·", '↔', '↓', '®', '™']

    for i in spaces:
        string = string.replace(i, ' ')
    return string


def sentence_split_stanford(tabst):
    """
    Sentence Splitting Using Stanford sentence splitter
    """
    split_lines = []
    for s in tabst:
        annotated = nlp.annotate(s, properties_ss)
        for sentence in annotated['sentences']:
            text = ' '.join([t['originalText'] for t in sentence['tokens']])
            split_lines.append(text)
    return split_lines


def tokenize_stanford(sents):
    token_sents = []
    for i, s in enumerate(sents):
        annotated = nlp.annotate(s, properties_tok)
        tokens = [t['originalText'] for t in annotated['tokens']]

        text = []
        for j, t in enumerate(tokens):
            if t == "'s":
                text.append(t)
            elif t == "''":
                text.append(t)
            else:
                text.append(t.replace("'", " ' "))

        text = ' '.join(text)

        text = text.replace("-LRB-", '(')
        text = text.replace("-RRB-", ')')
        text = text.replace("-LSB-", '[')
        text = text.replace("-RSB-", ']')
        text = text.replace("-LCB-", '{')
        text = text.replace("-RCB-", '}')
        text = text.replace("'s", " 's")
        text = text.replace('-', ' - ')
        text = text.replace('/', ' / ')
        text = text.replace('+', ' + ')
        text = text.replace('.', ' . ')
        text = text.replace('. .', ' . ')
        text = text.replace('=', ' = ')
        text = text.replace('*', ' * ')
        text = text.replace('\xa0', " ")
        text = re.sub(' +', ' ', text).strip()  # remove continuous spaces

        token_sents.append(text)
    return token_sents


def sentence_split_genia(tabst):
    """
    Sentence Splitting Using GENIA sentence splitter
    Args:
        tabst: (list) title+abstract

    Return
        (list) all sentences in abstract
    """
    os.chdir(genia_splitter)

    with open('temp_file.txt', 'w') as ofile:
        ofile.write(''.join(tabst))
    os.system('./geniass temp_file.txt temp_file.split.txt > /dev/null 2>&1')

    split_lines = []
    with open('temp_file.split.txt', 'r') as ifile:
        for line in ifile:
            line = line.rstrip()
            if line != '':
                split_lines.append(line.rstrip())
    os.system('rm temp_file.txt temp_file.split.txt')
    os.chdir(pwd)
    return split_lines


def tokenize_genia(sents):
    """
    Tokenization using Genia Tokenizer
    Args:
        (list) sentences
    Return:
        (list) tokenized sentences
    """
    token_sents = []
    for i, s in enumerate(sents):
        s = s.replace("``", "''")
        s = s.replace("`s", "'s")

        tokens = []
        for word, base_form, pos_tag, chunk, named_entity in tagger.tag(s):
            tokens += [word]

        text = []
        for t in tokens:
            if t == "'s":
                text.append(t)
            elif t == "''":
                text.append(t)
            else:
                text.append(t.replace("'", " ' "))

        text = ' '.join(text)
        text = text.replace("-LRB-", '(')
        text = text.replace("-RRB-", ')')
        text = text.replace("-LSB-", '[')
        text = text.replace("-RSB-", ']')
        text = text.replace("``", '"')
        text = text.replace("`", "'")
        text = text.replace("'s", " 's")
        text = text.replace('-', ' - ')
        text = text.replace('/', ' / ')
        text = text.replace('+', ' + ')
        text = text.replace('.', ' . ')
        text = text.replace('=', ' = ')
        text = text.replace('*', ' * ')

        text = re.sub(' +', ' ', text).strip()  # remove continuous spaces

        if "''" in ''.join(s):
            pass
        else:
            text = text.replace("''", '"')

        token_sents.append(text)
    return token_sents
