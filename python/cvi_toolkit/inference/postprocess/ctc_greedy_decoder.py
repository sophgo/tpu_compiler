#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from argparse import ArgumentParser
from os.path import join
import argparse
import sys
import codecs


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz_file",
        help="npz input file, with lstm network result"
    )
    parser.add_argument(
        "--tensor",
        help="output_tensor in npz file"
    )
    parser.add_argument(
        "--label_file",
        help="label file"
    )
    parser.add_argument(
        "--encoding", type=str, default='gb18030',
        help="label file encoding"
    )
    parser.add_argument(
        "--output",
        type=str, default='',
        help="output txt file, if none, will print directly"
    )
    return parser


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    blobs = np.load(args.npz_file)
    data = blobs[args.tensor]
    merge_repeated = True
    blank_index = 0
    T_ = data.shape[0]
    N_ = data.shape[1]
    data_max = np.argmax(data, 2)
    data_max = data_max.reshape((T_, N_))
    data_max = np.transpose(data_max, (1, 0))
    output = np.full((N_, T_), int(-1))
    for n in range(N_):
        prev_class_idx = -1
        index = 0
        for t in range(0, T_):
            max_class_idx = data_max[n][t]
            repeated = (merge_repeated and max_class_idx == prev_class_idx)
            if max_class_idx != blank_index and not repeated:
                output[n][index] = int(max_class_idx)
                index += 1
                prev_class_idx = max_class_idx
    labels = codecs.open(args.label_file, 'r', encoding=args.encoding)
    ss = labels.readlines()
    o_f = None
    if args.output != "":
        o_f = codecs.open(args.output, 'w', encoding='utf-8')
    else:
        sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    text = ""
    for n in range(N_):
        for t in range(T_):
            index = output[n][t]
            char = ss[index].strip()
            if index != -1:
                text += char
            else:
                break
        text += '\r\n'
    if o_f != None:
        o_f.write(text)
        o_f.flush()
        o_f.close()
    else:
        print('ctc result: ')
        print(text)
    labels.close()
