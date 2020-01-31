#!/usr/bin/env bash

python3 process.py --input_folder ../data/ACE-2005/brat/train/ \
                   --output_file ../data/ACE-2005/train \
                   --domain gen \
                   --processed \
                   --level doc

python3 process.py --input_folder ../data/ACE-2005/brat/dev/ \
                   --output_file ../data/ACE-2005/dev \
                   --domain gen \
                   --processed \
                   --level doc

python3 process.py --input_folder ../data/ACE-2005/brat/test/ \
                   --output_file ../data/ACE-2005/test \
                   --domain gen \
                   --processed \
                   --level doc

cd ..
