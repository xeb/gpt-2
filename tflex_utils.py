import tqdm
import sys
try:
  from tensorflow import io
  gfile = io.gfile
except:
  from tensorflow import gfile
import time

def isfile(f):
  return gfile.exists(f) and not gfile.isdir(f)

def exists(f):
  return gfile.exists(f)

def isdir(f):
  return gfile.isdir(f)

def glob(patterns):
  if isinstance(patterns, str):
    patterns = patterns.split(',')
  paths = []
  for pattern in patterns:
    paths.extend(gfile.glob(pattern))
  return paths

def file_size(f):
  if isinstance(f, str):
    with try_open(f) as f:
      return file_size(f)
  if isinstance(f, gfile.GFile):
    return f.size()
  else:
    was = f.tell()
    try:
      f.seek(0, 2)
      pos = f.tell()
    finally:
      f.seek(was, 0)
    return pos

def count_lines(f, verbose=True, ignore_errors=True):
    if isinstance(f, str):
      with try_open(f) as f:
        return count_lines(f)
    n = 0
    prev = None
    size = file_size(f)
    f.seek(0)
    prev_pos = 0
    pos = 0
    update_time = time.time() + 1.0
    with tqdm.tqdm(total=size, desc="Counting lines in text file...") as pbar:
      while True:
        try:
          for line in f:
            pos += len(line)
            if time.time() > update_time:
              pbar.update(pos - prev_pos)
              prev_pos = pos
              update_time = time.time() + 1.0
            n += 1
            prev = line
            #print(n)
          break
        except UnicodeDecodeError:
          if verbose:
            sys.stderr.write('Error on line %d after %s\n' % (n+1, repr(prev)))
          if not ignore_errors:
            raise
    # Reset the file iterator, or later calls to f.tell will
    # raise an IOError or OSError:
    f.seek(0)
    return n

def for_each_line(f, total=None, verbose=True, ignore_errors=True, message=None):
    if isinstance(f, str):
      with try_open(f) as infile:
        for i, line in for_each_line(infile, total=total, verbose=verbose, ignore_errors=ignore_errors, message=message):
          yield i, line
    elif isinstance(f, list):
      i = 0
      for line in tqdm.tqdm(f) if verbose else f:
        yield i, line
        i += 1
    else:
      i = 0
      prev = None
      size = file_size(f)
      pos = 0
      prev_pos = 0
      n = 0
      while True:
        try:
          with tqdm.tqdm(total=size) as pbar:
            for line in f:
              yield i, line
              i += 1
              pos += len(line)
              pbar.update(pos - prev_pos)
              prev = line
              prev_pos = pos
            break
        except UnicodeDecodeError:
          n += 1
          if verbose:
            sys.stderr.write('Error on line %d after %s\n' % (i+n+1, repr(prev)))
          if not ignore_errors:
            raise

import time

def try_open(filename, *args, **kws):
  if filename.startswith("gs://"):
    return gfile.GFile(filename, *args, **kws)
  else:
    return open(filename, *args, **kws)

def ensure_open(filename, *args, **kws):
  while True:
    try:
      return try_open(filename, *args, **kws)
    except Exception as exc:
      if not isinstance(exc, FileNotFoundError):
        import traceback
        traceback.print_exc()
      sys.stderr.write('Failed to open file: %s. Trying again in 1.0 seconds...\n' % filename)
      time.sleep(1.0)

import numpy as np
import io

def tokens_to_buffer(chunks, stride):
  assert stride in [2, 4]
  tokens = np.array(chunks, dtype=np.uint16 if stride == 2 else np.int32)
  return tokens.tobytes()

def tokens_from_buffer(data, stride):
  assert stride in [2, 4]
  return np.frombuffer(data, dtype=np.uint16 if stride == 2 else np.int32)

def tokens_to_file(out, chunks, stride):
  if isinstance(out, gfile.GFile):
    data = tokens_to_buffer(chunks, stride)
    out.write(data)
  else:
    assert stride in [2, 4]
    tokens = np.array(chunks, dtype=np.uint16 if stride == 2 else np.int32)
    tokens.tofile(out)

def tokens_from_file(f, stride):
  if isinstance(f, gfile.GFile):
    return tokens_from_buffer(f.read(), stride)
  if isinstance(f, str) and f.startswith('gs://'):
    with gfile.GFile(f, 'rb') as f:
      return tokens_from_file(f, stride)
  assert stride in [2, 4]
  return np.fromfile(f, dtype=np.uint16 if stride == 2 else np.int32)

