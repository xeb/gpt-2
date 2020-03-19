#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./decode.py /path/to/input.npz

import argparse
import numpy as np

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]

import encoder
from load_dataset import load_dataset

import struct
import tqdm

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='CHARS', type=int, default=50000, help='Concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--append', action='store_true', help='append to the output file?')
parser.add_argument('in_npz', metavar='IN.npz', type=str, help='Input file path')
parser.add_argument('out_tok', metavar='OUT.tok', type=str, default='-', nargs='?', help='Output file path')

from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    chunks = load_dataset(enc, args.in_npz, args.combine)
    for chunk in tqdm.tqdm(chunks):
      with (nullcontext() if args.out_tok == '-' else open(args.out_tok, "ab" if args.append else "wb")) as f:
        for i, token in enumerate(tqdm.tqdm(chunk)):
          if args.out_tok == '-' and i % 16 == 0 and i > 0:
            sys.stdout.write('\n')
            sys.stdout.flush()
          if args.out_tok == '-':
            sys.stdout.write('%d ' % token)
          else:
            f.write(struct.pack('i', token))

if __name__ == '__main__':
    main()
