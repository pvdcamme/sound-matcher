"""
  A program to find repeated audio fragments in an audio file.

  Very prototype at the moment.
"""
import struct
import bisect

import numpy as np
import matplotlib.pyplot as plt
import time

import cupy
import cupy.fft
import cupyx.scipy.signal

import decimator

FILE_NAME = "stubru.raw"

def benchmark_np():
  size = 2 ** 18
  cnt = 128
  to_check = np.random.random(size)
  others = [np.random.random(size) for _ in range(cnt)]
  total_max = 0
  start = time.time()
  for o in others:
    current_max = np.max(np.fft.irfft(to_check * o))
    total_max = max(total_max, current_max)

  end = time.time()
  print(f"Numpy: {total_max} in {1e3 * (end - start):.2f} ms")


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
    
def chunked_read(name, size):
    offset = 0
    while True:
        data = cupy.fromfile(name, dtype=np.int16, count=size, offset=offset)
        if len(data) != size:
            return
        offset += size
        yield normalize_array(cupy.asarray(data, dtype=np.float32))

def chunked_read_cpu(name, size):
    offset = 0
    while True:
        data = np.fromfile(name, dtype=np.int16, count=size, offset=offset)
        if len(data) != size:
            return
        offset += size
        yield normalize_array(np.asarray(data, dtype=np.float32))



def counter(start = 0):
    while True:
        yield start
        start += 1
        
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
            
def find_most_popular_section3(length=2**18, name=FILE_NAME):
    def calculate_hash(sorted_idx, moment):
      seen = set(range(4))
      current_hash = []
      for peak_idx in map(int, reversed(sorted_idx)):
        if peak_idx not in seen:
          seen = seen.union(range(peak_idx - 10, peak_idx + 10))
          current_hash.append(peak_idx)
        if len(current_hash) >= 10:
          break
      current_hash.append(moment)
      return current_hash

      
    fun_start = time.time()

    small_chunk = 2**16 #sample_rate

    chunk_hashes = []
    start_moment =time.time()
    stream = cupy.cuda.Stream(non_blocking=True)
    stream.use()
    sorted_elems = []

    together = 8
    for idx, part in enumerate(chunked_read(name, together * small_chunk), start =1):
      sample_start= idx * together * small_chunk
      if idx % 1024 == 0:
        print(f"{together * idx / (time.time() - start_moment):.2f} part/s -- {len(chunk_hashes)} hashes -- {sample_start/ 44100:.2f}s")

      for inner_idx in range(together):
        current_sample_start = (sample_start + inner_idx * small_chunk) / 44100
        bb = normalize_array(part[inner_idx * small_chunk: (inner_idx + 1) * small_chunk])
        bb = cupy.fft.rfft(part)
        max_val = cupy.abs(bb)
        sorted_idx = cupy.argsort(max_val)
        sorted_elems.append((sorted_idx, current_sample_start))
        while len(sorted_elems) > 16:
          chunk_hashes.append(calculate_hash(*sorted_elems[0]))
          del sorted_elems[0]
    for sorted_el in sorted_elems:
      chunk_hashes.append(calculate_hash(*sorted_el))

    chunk_hashes = sorted(chunk_hashes)
    for pp in chunk_hashes[:10]:
      print(f"Hash: {pp}")
 
def read_chunks_gpu():
    start = time.time()
    name = "stubru.raw"
    chunk_size = 2 * 1024 * 1024
    ctr = 0
    offset = 0
    while True:
        data = cupy.fromfile(name, dtype=np.int16, count=chunk_size, offset=offset)
        cupy.fft.rfft(data)
        offset += len(data) * 2
        ctr += 1

        if len(data) < chunk_size:
            print(f"Too small: {len(data)} vs {chunk_size}")
            break

    print(f"Total time: {time.time() - start}")            
    return ctr

def show_match():
  data = np.fromfile("stubru.raw", dtype=np.int16, count=  16 * 1024 * 1024)
  data = normalize_array(np.asarray(data, dtype=np.double()))
  timing = np.arange(len(data)) / 44100.

  figure = plt.gcf()

  graph1 = figure.add_subplot(211)
  graph1.plot(timing, data)
  x_axis = plt.gca().get_xaxis()


  graph2 = figure.add_subplot(212)
  graph2.psd(data, Fs=44100, NFFT=256)

  view_interval = [0, 0]
  def onclick(event):
      global view_interval
      new_interval = list(x_axis.get_view_interval())
      if new_interval == view_interval:
          return
      view_interval = new_interval
      a, b = view_interval

      start_moment = time.time()
      start = bisect.bisect(timing, a)
      end = bisect.bisect(timing, b)

      graph2.clear()
      graph2.psd(data[start:end], Fs=44100, NFFT=4096)
    
      print(f"{view_interval} -> {(start, end)} in {time.time() - start_moment:.2}")
    

  cid = plt.gcf().canvas.mpl_connect('motion_notify_event', onclick)    
  plt.show()

find_most_popular_section3(2 ** 18)

#decimator.plot_filter_response(decimator.low_pass_filter)

#example_vals = chunked_read(FILE_NAME, 4 * 1024 * 1024)
#decimator.decimate(next(example_vals))
