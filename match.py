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
      Splits up the file in multiple hashes of length.
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


    with stream:
      for idx, part in enumerate(chunked_read(name, length), start=0):
          sample_start = idx * length 
          if idx % 1024 == 0:
            idx_speed = idx / (time.time() - start_moment)
            msg = f"{idx_speed:.2f} part/s -- {len(chunk_hashes)} hashes -- {sample_start/ 44100:.2f}s"
            print(msg)

          bb = cupy.fft.rfft(part)
          max_val = cupy.abs(bb)
          sorted_idx = cupy.argsort(max_val)
          chunk_hashes.append(calculate_hash(sorted_idx, sample_start))
          stream.synchronize()
    return chunk_hashes

def create_hashes_parallel(length=2**16, name=FILE_NAME):
    """
      Splits up the file in multiple hashes of length.
    """
    TOGETHER = 128
    def calculate_hash(sorted_idx, moment):
        seen = set(range(4))
        current_hash = []
        for peak_idx in reversed(sorted_idx):
            if peak_idx not in seen:
                current_hash.append(peak_idx)
                if len(current_hash) == 10:
                    break
                seen = seen.union(range(peak_idx - 10, peak_idx + 10))
        return (current_hash, moment)

    chunk_hashes = []
    start_moment = time.time()
    stream = cupy.cuda.Stream(non_blocking=True)
    stream_other = cupy.cuda.Stream(non_blocking=True)

    with stream:
      previous= None
      for idx, part in enumerate(chunked_read(name, TOGETHER * length, start=0)):
          it_start = time.time()
          rows = part.reshape((TOGETHER, length))
          max_val = rows.max(axis=1)
          min_val = rows.min(axis=1)

          scale = max_val - min_val
          rows = ((rows.T- (min_val + scale / 2)) * (2. / scale)).T

          bb = cupy.fft.rfft(rows, axis=1)
          magnitudes = cupy.abs(bb)

          starts = [length * (idx * TOGETHER + indiv_idx) for indiv_idx in range(TOGETHER)]

          sorted_idx = magnitudes.argsort(axis=1)
          sorted_idx = cupy.asnumpy(sorted_idx)

          gpu_done = time.time()

          for row_idx in range(TOGETHER):
            chunk_hashes.append(calculate_hash(sorted_idx[row_idx,:], starts[row_idx]))

          if idx % 8 == 0:
            idx_speed =  len(chunk_hashes) / (time.time() - start_moment)
            msg = f"{idx_speed:.2f} part/s -- {len(chunk_hashes)} hashes -- {starts[0]/ 44100:.2f}s -- GPU took: {gpu_done - it_start} + cpu= {time.time() - gpu_done}"
            print(msg)


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

print("Starting")
chunks = create_hashes_parallel(2**16)
start, closest_results, end = (time.time(), calc_closest_pair_gpu(chunks),
                               time.time())
print(f"GPU Closest in {len(chunks) * len(chunks) / (end - start)} assoc/sec")

decent_to_show =[a for a in closest_results if 0.1 < a[0] <= 1]
np.random.shuffle(decent_to_show)
closest = decent_to_show[0]
show_match(FILE_NAME, closest[1], closest[2], 2**16)
sys.exit(0)

def multiple_together(vals):
  stream = cupy.cuda.Stream(non_blocking=True)
  with stream:
    vals = vals.reshape((together_count, 2**12))
    fft = cupy.fft.rfft(vals, axis=1)
    fft = cupy.abs(fft)
    fft[:,0] = 0
    max_idx = cupy.argmax(fft, axis=1)
  stream.synchronize()
  return max_idx.get()

def multiple_together_t(vals):
  stream = cupy.cuda.Stream(non_blocking=True)
  with stream:
    vals = vals.reshape((2 ** 12, together_count))
    fft = cupy.fft.rfft(vals, axis=0)
    fft = cupy.abs(fft)
    fft[0,:] = 0
    max_idx = cupy.argmax(fft, axis=0)
  stream.synchronize()
  return max_idx.get()

def multiple_seperate(vals):
  result = []
  stream = cupy.cuda.Stream(non_blocking=True)
  with stream:
    for idx in range(together_count):
      row = cupy.fft.rfft(vals[idx * 2**12: (idx+1) * 2**12])
      row = cupy.abs(row)
      row[0] = 0
      max_idx = cupy.argmax(row)
      result.append(max_idx)
  return [float(val) for val in result]


print("starting benchmark")

exponents = []
gpu_together= []

gpu_together_t= []
gpu_seperate= []


cpu_together= []
cpu_seperate= []



for exp in sorted(set([int(2 ** exp) for exp in np.arange(1, 10, 0.1)])):
  together_count = exp
  print(f"Testing {together_count}")
  exponents.append(together_count)
  vals = cupy.random.random(together_count * 2** 12, dtype=cupy.float32)

  res = cupyx.profiler.benchmark(multiple_together, (vals,), n_repeat=100, n_warmup = 20)
  gpu_together.append(1E6 * np.average(res.gpu_times) / together_count)
  cpu_together.append(1E6 * np.average(res.cpu_times) / together_count)
  
  res = cupyx.profiler.benchmark(multiple_together_t, (vals,), n_repeat=100, n_warmup = 20)
  gpu_together_t.append(1E6 * np.average(res.gpu_times) / together_count)
 

  res = cupyx.profiler.benchmark(multiple_seperate, (vals,), n_repeat=100, n_warmup = 20)
  gpu_seperate.append(1E6 * np.average(res.gpu_times) / together_count)
  cpu_seperate.append(1E6 * np.average(res.cpu_times) / together_count)
  
plt.plot(exponents, gpu_together, label="together")
plt.plot(exponents, gpu_together_t, label="together_t")
plt.plot(exponents, gpu_seperate, label="separate")
plt.legend()
plt.grid()
plt.xscale("log")
plt.ylabel("duration [us]")

plt.show()

def snr_vs_duration(file_name):
  duration = 1024
  durations = [] 
  avg_snr = []
  min_snr = []
  max_snr = []

  stream = cupy.cuda.Stream(non_blocking=True)
  while duration < 1024 * 1024:
    start = time.time()
    snrs = []
    with stream:
      interesting = (10e3 / 44100)
      for idx, part in zip(range(8 * 1024), chunked_read(file_name, duration)):
        fft = cupy.abs(cupy.fft.rfft(part))
        fft = fft[0:int(duration * interesting)]
        largest = cupy.max(fft)
        avg = cupy.median(fft)
        snrs.append(largest / avg)

    snrs = np.array([float(val) for val in snrs])
    durations.append(duration / 44100.)
    avg_snr.append(np.quantile(snrs,0.5))
    min_snr.append(np.quantile(snrs,0.2))
    max_snr.append(np.quantile(snrs, 0.8))

    print(f" {duration} :: {time.time() - start}")
    duration = (int(1.1 * duration) >> 3) << 3



  plt.plot(durations, avg_snr)
  plt.plot(durations, max_snr)
  plt.plot(durations, min_snr)
  plt.show()

#snr_vs_duration(FILE_NAME)
    
