#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 11/09/2019

author: fenia
"""

import os, sys


with open(sys.argv[1], 'r') as infile:
    epoch = -1
    for line in infile:
        line = line.rstrip()
        if line.startswith('Best epoch:'):
            epoch = line.split('Best epoch:')[1]
            epoch = int(epoch)
        else:
            continue
    print(epoch)


