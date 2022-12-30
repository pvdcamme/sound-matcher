"""
  A program to find repeated audio fragments in an audio file.

  Very prototype at the moment.
"""
import sys
import struct
import bisect
import itertools as it

import numpy as np
import matplotlib.pyplot as plt
import time

import cupy
import cupy.fft
import cupyx.scipy.signal

import decimator

FILE_NAME = "stubru.raw"


def read_chunks():
    start = time.time()
    name = "stubru.raw"
    chunk_size = 16 * 1024 * 1024
    ctr = 0
    offset = 0
    while True:
        data = np.fromfile(name,
                           dtype=np.int16,
                           count=chunk_size,
                           offset=offset)
        np.fft.rfft(data)
        offset += len(data) * 2
        ctr += 1
        if len(data) < chunk_size:
            print(f"Too small: {len(data)} vs {chunk_size}")
            break

    print(f"Total time: {time.time() - start}")
    return ctr


def normalize_array(data):
    min_val = data.min()
    max_val = data.max()
    scale = max_val - min_val
    data = (data - (min_val + scale / 2)) * (2. / scale)
    return data


def counter(start=0):
    while True:
        yield start
        start += 1


def chunked_read(file_name, size, start=0):
    """
      Iterates over the data within file_name returns them in
      chunks of size.
    """
    offset = start * 2
    while True:
        data = cupy.fromfile(file_name,
                             dtype=np.int16,
                             count=size,
                             offset=offset)
        if len(data) != size:
            return
        offset += 2 * size
        yield normalize_array(cupy.asarray(data, dtype=np.float32))


def create_hashes(length=2**16, name=FILE_NAME):
    """
        
    """
    def calculate_hash(sorted_idx, moment):
        seen = set(range(4))
        current_hash = []
        for peak_idx in reversed(sorted_idx.get()):
            if peak_idx not in seen:
                seen = seen.union(set(range(peak_idx - 10, peak_idx + 10)))
                current_hash.append(peak_idx)
            if len(current_hash) >= 10:
                break
        return (current_hash, moment)

    chunk_hashes = []
    start_moment = time.time()
    stream = cupy.cuda.Stream(non_blocking=True)

    for idx, part in enumerate(chunked_read(name, length), start=0):
        sample_start = idx * length 
        if idx % 1024 == 0:
          idx_speed = idx / (time.time() - start_moment)
          msg = f"{idx_speed:.2f} part/s -- {len(chunk_hashes)} hashes -- {sample_start/ 44100:.2f}s"
          print(msg)

        with stream:
          bb = cupy.fft.rfft(part)
          max_val = cupy.abs(bb)
          sorted_idx = cupy.argsort(max_val)
          chunk_hashes.append(calculate_hash(sorted_idx, sample_start))
    return chunk_hashes


def show_match(file_name, start_idx, end_idx, length):
    start_data = next(chunked_read(file_name, length, start_idx * length))
    compare_data = next(chunked_read(file_name, length, end_idx * length))

    timing = np.arange(len(start_data)) / 44100.
    closest_match = int(
        cupy.argmax(cupyx.scipy.signal.correlate(start_data, compare_data)))

    if closest_match > length / 2:
        closest_match -= length

    plt.plot(timing, start_data.get())
    plt.plot(timing + closest_match / 44100, compare_data.get())
    plt.show()


def calc_closest_pair_gpu(chunks):
    """
    Finds the closest matching other chunk from the list.
    These pairs should as similar as possible to each other.
    """
    start_fun = time.time()
    ll = len(chunks[0][0])
    base = cupy.ones((len(chunks), ll))
    weights = 0.8**cupy.arange(ll, dtype=cupy.float32)

    stream = cupy.cuda.Stream(non_blocking=True)

    for idx, (ch, _start) in enumerate(chunks):
        base[idx, :] = cupy.array(ch, dtype=cupy.float32)

    result = []
    with stream:
        difference = cupy.empty_like(base)
        for idx, _ in enumerate(chunks):
            cupy.subtract(base, base[idx, :], out=difference)
            cupy.abs(difference, out=difference)
            current = cupy.tensordot(difference, weights, axes=1)
            current[idx] = 1E9
            min_place = cupy.argmin(current)
            min_val = cupy.min(current)
            result.append((min_val, idx, min_place))

    return [(float(mm), int(idx_1), int(idx_2)) for mm, idx_1, idx_2 in result]


chunks = create_hashes(2**16)

start, closest_results, end = (time.time(), calc_closest_pair_gpu(chunks),
                               time.time())
print(f"GPU Closest in {len(chunks) * len(chunks) / (end - start)} assoc/sec")

