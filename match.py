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
        data = np.fromfile(name, dtype=np.int16, count=chunk_size, offset=offset)
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
    data = (data - (min_val + scale/2 )) * (2. /scale )
    return data

def counter(start = 0):
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
        data = cupy.fromfile(file_name, dtype=np.int16, count=size, offset=offset)
        if len(data) != size:
            return
        offset += 2 * size
        yield normalize_array(cupy.asarray(data, dtype=np.float32))
    
def find_most_popular_section(length=2**18):
    fun_start = time.time()
    name = "stubru.raw"
    small_chunk = length
    chunk_size = 4 * length #32 * 1024 * 1024
    cuda_streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(8)]
    sample_rate =44100
    peak_max = 0
    peak_position = 0
    
    for base_idx, bb in enumerate(chunked_read(name, length)):
        moment = base_idx * length / sample_rate
        print(f"Starting with new signal {moment}s")
        to_pad = chunk_size - length
        padded = cupy.pad(bb, to_pad // 2)
        assert len(padded) == chunk_size

        max_val = 0
        bb = cupy.conj(cupy.fft.fft(padded))
        start = time.time()

        ongoing_ops = []
        streams = chunked_read(name,chunk_size)            

        for idx in counter():
            while len(ongoing_ops) > 0  and ongoing_ops[0]["stream"].done:
                max_val = max(max_val, ongoing_ops[0]["array"].max())
                del ongoing_ops[0]
                
            current = cuda_streams[idx % len(cuda_streams)]
            if not current.done:
                print("Not ready")
                current.synchronize()
            current.use()
            data = next(streams, None)
            if data is None:
                break
            fft_larger = cupy.fft.fft(data)
            gg = bb * fft_larger
            d = cupy.fft.ifft(gg)
            ongoing_ops.append({"stream": current, "array": d})

        while len(ongoing_ops) > 0:        
            max_val = max(max_val, ongoing_ops[0]["array"].max())
            del ongoing_ops[0]
            
        cupy.cuda.Stream.null.use()        
        if abs(max_val) > abs(peak_max):
            peak_pos = moment
            peak_max = abs(max_val)
        print(f"chunk result {max_val} in {time.time() - start:.2f}, x{((base_idx + 1) * length / sample_rate)/(time.time() - fun_start):.2f} -- Best moment {peak_pos:.2f}")
       
def find_most_popular_section2(length=2**18, name=FILE_NAME):
    fun_start = time.time()
    sample_rate =44100

    small_chunk = length
    chunk_size = 4 * length
    cuda_streams = [cupy.cuda.Stream(non_blocking=True) for _ in range(4)]    
    base_stream = enumerate(chunked_read(name, length))
    num_base_sections = 128
    def pp():
        current = {"peak_pos": 0, "peak_max": 0}
        def process_done(together, do_sync=False):
            to_keep = []
            for stream, a_max, base_idx in together:                
                if not stream.done:
                    to_keep.append((stream, a_max, base_idx))
                    continue
                elif abs(a_max) > current["peak_max"]:
                    peak_pos = base_idx * length / sample_rate
                    peak_max = abs(a_max)
                    current["peak_pos"] = peak_pos
                    current["peak_max"] = peak_max
            together.clear()
            together.extend(to_keep)
            return (current["peak_pos"], current["peak_max"])
        return process_done
    peaker = pp()
    together = []   
    while True:
        base_selection = []
        for idx in range(num_base_sections):
            stream =cuda_streams[ idx % len(cuda_streams)]
            stream.use()
            a_next = next(base_stream, None)
            if a_next is None:
                break
            base_idx, base_sample = a_next
            to_pad = chunk_size - length
            padded = cupy.pad(base_sample, to_pad // 2)
            bb = cupy.conj(cupy.fft.rfft(padded))
            base_selection.append((base_idx, cupy.asarray(bb, dtype=cupy.complex64)))

        highest_idx = 0

        print("Processed batch, iterating over full")
        for fft_larger in map(cupy.fft.rfft, chunked_read(name,chunk_size)):

            start_batch = time.time()
            for stream_idx, (idx, smaller) in enumerate(base_selection):
                stream = cuda_streams[stream_idx % len(cuda_streams)]
                stream.use()

                inverse = cupy.fft.irfft(smaller * fft_larger)
                together.append((stream, inverse.max(), idx))

                highest_idx = max(highest_idx, idx)

            cuda_streams[0].synchronize()
            done_batch = time.time()
            print(f"Total: {1e3 * (done_batch - start_batch):.2f}")

        cuda_streams[0].synchronize()
        cupy.cuda.Stream.null.use()
        peak_pos, peak_max = peaker(together)
        moment = highest_idx * length / sample_rate

        print(f"chunk results up to moment, x{((base_idx + 1) * length / sample_rate)/(time.time() - fun_start):.2f} -- Best moment {peak_pos:.2f}-- {len(together)} running")
    return peaker(together, True)
            
def create_hashes(length=2**16, name=FILE_NAME):
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
      
    fun_start = time.time()
    small_chunk = length

    chunk_hashes = []
    start_moment =time.time()
    stream = cupy.cuda.Stream(non_blocking=True)
    stream.use()
    sorted_elems = []

    for idx, part in enumerate(chunked_read(name, small_chunk), start =0):
      
      sample_start= idx * small_chunk
      if idx % 1024 == 0:
        print(f"{ idx / (time.time() - start_moment):.2f} part/s -- {len(chunk_hashes)} hashes -- {sample_start/ 44100:.2f}s")

      bb = cupy.fft.rfft(part)
      max_val = cupy.abs(bb)
      sorted_idx = cupy.argsort(max_val)
      sorted_elems.append((sorted_idx, sample_start))
      while len(sorted_elems) > 12:
        chunk_hashes.append(calculate_hash(*sorted_elems[0]))
        del sorted_elems[0]
    for sorted_el in sorted_elems:
      chunk_hashes.append(calculate_hash(*sorted_el))
    return chunk_hashes

def show_match(file_name, start_idx, end_idx, length):
  start_data =  next(chunked_read(file_name, length, start_idx * length))
  compare_data = next(chunked_read(file_name, length, end_idx * length))

  timing = np.arange(len(start_data)) / 44100.
  closest_match = int(cupy.argmax(cupyx.scipy.signal.correlate(start_data, compare_data)))

  if closest_match > length / 2:
    closest_match -= length
  

  plt.plot(timing, start_data.get())
  plt.plot(timing + closest_match/44100, compare_data.get())
  plt.show()

def calc_closest_pair_gpu(chunks):
  """
    Finds the closest matching other chunk from the list.
    These pairs should as similar as possible to each other.
  """
  start_fun = time.time()
  ll = len(chunks[0][0])
  base = cupy.ones((len(chunks), ll))
  weights = 0.8 ** cupy.arange(ll, dtype=cupy.float32)

  stream = cupy.cuda.Stream(non_blocking=True)

  for idx, (ch, _start) in enumerate(chunks):
    base[idx, :] = cupy.array(ch, dtype=cupy.float32)

  result = []
  with stream:
    difference = cupy.empty_like(base)
    for idx, _ in enumerate(chunks):
      cupy.subtract(base, base[idx,:], out=difference)
      cupy.abs(difference, out=difference)
      current = cupy.tensordot(difference, weights, axes=1)
      current[idx] = 1E9
      min_place = cupy.argmin(current)
      min_val = cupy.min(current)
      result.append((min_val, idx, min_place))

  return [(float(mm), int(idx_1), int(idx_2)) for mm, idx_1, idx_2 in result]



chunks = create_hashes(2 ** 16)

start, closest_results, end =  (time.time(), calc_closest_pair_gpu(chunks), time.time())
print(f"GPU Closest in {len(chunks) * len(chunks) / (end - start)} assoc/sec")


first_chunk = next(chunked_read(FILE_NAME, 2**16, start =0))
next_chunk = next(chunked_read(FILE_NAME, 2**16, start =(2**16) * closest_results[0][2]))
plt.plot(cupy.asnumpy(first_chunk))
plt.plot(cupy.asnumpy(next_chunk))
plt.grid()
plt.show()


