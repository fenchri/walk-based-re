#!/usr/bin/python3

import numpy as np
import glob
import sys
from collections import OrderedDict
import argparse
from tqdm import tqdm
import subprocess

"""
Crop embeddings to the size of the dataset, i.e. keeping only existing words.
"""

def load_pretrained_embeddings(embeds, dim):
    """
        :param params: input parameters
        :returns
            dictionary with words (keys) and embeddings (values)
    """
    if embeds:
        result = subprocess.run(['wc', '-l', embeds], stdout=subprocess.PIPE)
        lens = int(result.stdout.decode('utf-8').split(' ')[0])
       
        E = OrderedDict()
        with open(embeds, 'r') as vectors:
            for x, line in tqdm(enumerate(vectors), desc='Loading embeddings', total=lens):
                if x == 0 and len(line.split()) == 2:
                    words, num = map(int, line.rstrip().split())
                else:
                    word = line.rstrip().split()[0]
                    vec = line.rstrip().split()[1:]
                    n = len(vec)
                    if len(vec) != dim:
                        # print('Wrong dimensionality: {} {} != {}'.format(word, len(vec), num))
                        continue
                    else:
                        E[word] = np.asarray(vec, dtype=np.float32)
        print('Pre-trained word embeddings: {} x {}'.format(len(E), n))
    else:
        E = OrderedDict()
        print('No pre-trained word embeddings loaded.')
    return E


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full_embeds', type=str)
    parser.add_argument('--out_embeds', type=str)
    parser.add_argument('--in_data', nargs='+')
    parser.add_argument('--dim', type=int)
    args = parser.parse_args()

    words = []
    print('\nExtracting words from the dataset ... ')

    for filef in args.in_data:
        with open(filef, 'r') as infile:
            for line in infile:
                line = line.strip().split('\t')[1]
                line = line.split('|')
                line = [l.split(' ') for l in line]
                line = [item for sublist in line for item in sublist]

                for l in line:
                    words.append(l)

    # make lowercase
    words_lower = list(map(lambda x:x.lower(), words))
    words = set(words)
    words_lower = set(words_lower)  # lowercased

    # Load embeddings
    embeddings = load_pretrained_embeddings(args.full_embeds, args.dim)  

    new_embeds = OrderedDict()
    for w in tqdm(embeddings.keys(), desc='Matching words'):
        if (w in words) or (w in words_lower):
            if w not in new_embeds:
                new_embeds[w] = embeddings[w]

    print('Writing final embeddings {} x {} ... '.format(len(new_embeds), args.dim), end="")
    with open(args.out_embeds, 'w') as outfile:
        for k, v in new_embeds.items():
            outfile.write('{} {}\n'.format(k, ' '.join(map(str, list(v)))))
    print('Done')

    coverage = 0
    for w in words:
        if w in new_embeds:
            coverage += 1
    print('Coverage (words in embeds/total words): {}/{}'.format(coverage, len(words)))


if __name__ == "__main__":
    main()
