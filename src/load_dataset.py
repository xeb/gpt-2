import glob
import numpy as np
import os
import tensorflow as tf
import tqdm
import sys
import tflex_utils

def load_dataset(enc, path, combine):
    paths = []
    if os.path.isfile(path):
        # Simple file
        paths.append(path)
    elif os.path.isdir(path):
        # Directory
        for (dirpath, _, fnames) in os.walk(path):
            for fname in fnames:
                paths.append(os.path.join(dirpath, fname))
    else:
        # Assume glob
        paths = glob.glob(path)

    token_chunks = []
    raw_text = ''
    for path in tqdm.tqdm(paths):
        if path.endswith('.npz'):
            # Pre-encoded
            with np.load(path) as npz:
                for item in npz.files:
                    token_chunks.append(npz[item])
        else:
            # Plain text
            with open(path, 'r') as fp:
                raw_text += fp.read()
            if len(raw_text) >= combine:
                tokens = np.stack(enc.encode(raw_text))
                token_chunks.append(tokens)
                raw_text = ''
            else:
                raw_text += '<|endoftext|>'
    if raw_text:
        tokens = np.stack(enc.encode(raw_text))
        token_chunks.append(tokens)
    return token_chunks


def binary_search(f, lo, hi):
    if f(lo) or not f(hi):
        return None
    while hi > lo + 1:
        mid = (lo + hi) // 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return hi


class Sampler(object):
    """Fairly samples a slice from a set of variable sized chunks.

    'Fairly' means that the distribution is the same as sampling from one concatenated chunk,
    but without crossing chunk boundaries."""

    def __init__(self, chunks, seed=None):
        self.chunks = chunks
        self.total_size = sum(chunk.shape[0] for chunk in chunks)
        self.boundaries = [0]
        for i in range(len(chunks)):
            self.boundaries.append(self.boundaries[-1] + chunks[i].shape[0])
        self.rs = np.random.RandomState(seed=seed)

    def sample(self, length):
        assert length < self.total_size // len(
            self.chunks
        ), "Dataset files are too small to sample {} tokens at a time".format(
            length)
        while True:
            index = self.rs.randint(0, self.total_size - length - 1)
            i = binary_search(lambda j: self.boundaries[j] > index, 0,
                              len(self.boundaries) - 1) - 1
            if self.boundaries[i + 1] > index + length:
                within_chunk = index - self.boundaries[i]
                return self.chunks[i][within_chunk:within_chunk + length]

def contbyte(b):
  n = ord(b)
  # https://en.wikipedia.org/wiki/UTF-8#Description
  return n & (1 << 7) and not n & (1 << 6)

def nextchar(f):
  b = f.read(1)
  # skip leading continuation bytes
  while b and contbyte(b):
    b = f.read(1)
  # append up to three continuation bytes
  for _ in range(3):
    c = f.read(1)
    if c and contbyte(c):
      b += c
    elif c:
      # not a continuation byte; back up one character
      f.seek(-1, 1)
      break
  return b.decode('utf-8')

def nextchars(f, n):
  s = []
  i = 0
  while i < n:
    c = nextchar(f)
    if c is None:
      break
    if c != u'\r':
      s.append(c)
      i += 1
  if len(s) > 0:
    return ''.join(s)

def grab_tokens(f, enc, n):
  n += 4
  count = n
  line = nextchars(f, count)
  if not line:
    return [], None
  tokens = enc.encode(line)
  while len(tokens) < n:
    count *= 2
    l = nextchars(f, count)
    if not l:
      break
    line += l
    tokens = enc.encode(line)
  # skip the first couple tokens in case we started in the middle of a
  # word (which is the likely case for a random seek anywhere in the
  # dataset).
  return tokens[3:], line

import threading

class TextSampler(object):
  def __init__(self, fp, enc, seed=None, verbose=False, use_locking=False):
    if isinstance(fp, str):
      fp = open(fp, 'rb')
    self.fp = fp
    fp.seek(0, 2)
    self.total_size = fp.tell()
    self.rs = np.random.RandomState(seed=seed)
    self.enc = enc
    self.verbose = verbose
    self.lock = threading.Lock() if use_locking else None

  def sample(self, length):
    try:
      if self.lock:
        self.lock.acquire()
      attempts = 0
      while True:
        attempts += 1
        if attempts > 10:
          print('Could not sample from dataset; too small?')
          return None
        index = self.rs.randint(0, self.total_size)
        self.fp.seek(index, 0)
        tokens, line = grab_tokens(self.fp, self.enc, length)
        if len(tokens) >= length:
          if self.verbose:
            line = self.enc.decode(tokens)
            print(repr(line))
          return tokens[0:length]
    finally:
      if self.lock:
        self.lock.release()

import struct
import os
import time

class TokenSampler(object):
  def __init__(self, infile, enc=None, seed=None, verbose=None, half=False, use_locking=False):
    if verbose is None:
      verbose = 'VERBOSE' in os.environ
    assert isinstance(infile, str)
    self.infile = infile
    #if isinstance(fp, str):
    #  fp = open(fp, 'rb')
    self.fp = None
    self.enc = enc
    self.half = half
    self.stride = 2 if half else 4
    self.rs = np.random.RandomState(seed=seed)
    self.verbose = verbose
    self.lock = threading.RLock() if use_locking else None
    self.last_total_size = -1
    self.last_time = time.time()
    assert self.total_size() % self.stride == 0

  def close_unsafe(self):
    if self.fp is None:
      return
    #if self.verbose:
    #  print('Closing token file: ' + self.infile)
    self.fp.close()
    self.fp = None

  def total_size_unsafe(self):
    if time.time() - self.last_time > 5.0:
      self.close_unsafe()
    if self.fp is None:
      #print('Opening token file: ' + self.infile)
      self.fp = tflex_utils.ensure_open(self.infile, 'rb')
    self.fp.seek(0, 2)
    total_size = self.fp.tell()
    if self.verbose and self.last_total_size != total_size:
      sys.stderr.write('%d tokens in token file %s\n' % (total_size // self.stride, self.infile))
      self.last_total_size = total_size
    return total_size

  def sample_unsafe(self, length, max_attempts=-1):
    attempts = 0
    while True:
      attempts += 1
      if max_attempts >= 0 and attempts >= max_attempts:
        print('Could not sample from dataset; too small?')
        return None
      index = self.rs.randint(0, max(1, self.total_size_unsafe() // self.stride - length))
      self.fp.seek(index*self.stride, 0)
      data = self.fp.read(self.stride * length)
      n = len(data) // self.stride
      tokens = struct.unpack('<' + n*('H' if self.stride == 2 else 'i'), data)
      if len(tokens) < length:
        self.close_unsafe()
        sys.stderr.write('Dataset too small; waiting 1.0 seconds for at least %d tokens in the token file...\n' % length)
        time.sleep(1.0)
        continue
      if self.verbose and self.enc:
        line = self.enc.decode(tokens)
        print(repr(line[0:100] + '...'))
      return tokens

  def total_size(self):
    if self.lock:
      with self.lock:
        return self.total_size_unsafe()
    else:
      return self.total_size_unsafe()

  def close(self):
    if self.lock:
      with self.lock:
        return self.close_unsafe()
    else:
      return self.close_unsafe()

  def sample(self, length, max_attempts=-1):
    if self.lock:
      with self.lock:
        return self.sample_unsafe(length=length, max_attempts=max_attempts)
    else:
      return self.sample_unsafe(length=length, max_attempts=max_attempts)

class TokenStreamer(object):
  def __init__(self, fp, enc, use_locking=False):
    self.fp = fp
    self.line_count = len(fp) if isinstance(fp, list) else tflex_utils.count_lines(fp)
    self.enc = enc
    self.lock = threading.Lock() if use_locking else None

  def stream(self, step=128*1024, verbose=True, **kws):
    try:
      if self.lock:
        self.lock.acquire()
      start = time.time()
      total = 0
      for i, line in tflex_utils.for_each_line(self.fp, total=self.line_count, verbose=verbose, **kws):
        #import pdb; pdb.set_trace()
        tokens = self.enc.encode(line) if isinstance(line, str) else line
        yield tokens
        total += len(tokens)
        if i % step == 0 and verbose:
          elapsed = time.time() - start
          sys.stderr.write('%d tokens in %.4fs (%.4f tokens/sec)\n' % (total, elapsed, total/elapsed))
    finally:
      if self.lock:
        self.lock.release()

