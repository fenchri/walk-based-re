#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 04/06/2019

author: fenia
"""


import os, re, sys
import numpy as np
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str)
args = parser.parse_args()


with open(args.data_file, 'r') as infile:
    temp = [line for line in infile]

random.shuffle(temp)


with open('train.data', 'w') as train_out, open('dev.data', 'w') as dev_out:
    for n, l in enumerate(temp):
        
        if n < 800:
            dev_out.write(l)
        else:
            train_out.write(l)



