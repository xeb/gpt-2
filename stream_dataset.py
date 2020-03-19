#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./decode.py /path/to/input.npz

import argparse
import numpy as np

import os
import sys
sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src')]

import encoder
from load_dataset import load_dataset, TokenStreamer

import struct
import tqdm
import time
import tflex_utils

parser = argparse.ArgumentParser(
    description='Pre-encode text files into tokenized training set.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_name', metavar='MODEL', type=str, default='117M', help='Pretrained model name')
parser.add_argument('--combine', metavar='chars', type=int, default=50000, help='concatenate files with <|endoftext|> separator into chunks of this minimum size')
parser.add_argument('--append', action='store_true', help='append to the output file?')
parser.add_argument('--reopen_every', type=float, default=1.0, help='Reopen output file every N seconds')
parser.add_argument('in_npz', metavar='IN.npz', type=str, help='Input file path')
parser.add_argument('out_tok', metavar='OUT.tok', type=str, default='-', nargs='?', help='Output file path')

from contextlib import contextmanager

@contextmanager
def nullcontext(enter_result=None):
    yield enter_result

def main():
    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name)
    chunks = load_dataset(enc, args.in_npz, args.combine) if args.in_npz.endswith('.npz') else args.in_npz
    streamer = TokenStreamer(chunks, enc=enc)
    text_mode = not isinstance(chunks, list)
    total_size = sum([len(x) for x in chunks]) if not text_mode else streamer.line_count
    assert args.out_tok.endswith('.tok16') or args.out_tok.endswith('.tok32')
    half = args.out_tok.endswith('.tok16')
    desc = ("Required filesize: %.2f MB" % ((2 if half else 4) * total_size / 1024 / 1024)) if not text_mode else None
    reopen_time = time.time()
    with (tqdm.tqdm(ncols=100, desc=desc, total=total_size, unit_scale=True) if not text_mode else nullcontext()) as pbar:
      with (nullcontext() if args.out_tok == '-' else open(args.out_tok, "ab" if args.append else "wb")) as f:
        i = 0
        for chunk in streamer.stream():
          #import pdb; pdb.set_trace()
          for token in chunk:
            if i % 1024 == 0 and i > 0:
              if args.out_tok == '-':
                sys.stdout.write('\n')
                sys.stdout.flush()
              if f is not None:
                f.flush()
            if f is not None:
              if time.time() - reopen_time > args.reopen_every:
                f.close()
                f = tflex_utils.ensure_open(args.out_tok, "ab")
                reopen_time = time.time()
            i += 1
            if args.out_tok == '-':
              sys.stdout.write('%d ' % token)
            else:
              if half:
                assert token >= 0 and token < 65536
              else:
                pass # TODO: assert integer limits
              f.write(struct.pack('<H' if half else '<i', token))
            if not text_mode:
              pbar.update(1)

if __name__ == '__main__':
    main()
