"""
  Utils to reduce the sample rate of the audio.
  This helps to speed up the other analysis.
"""
import numpy as np
import matplotlib.pyplot as plt
import time

import cupy
import cupyx.scipy.signal


# Transition at 2kHz for a 44100 sample rate
low_pass_filter = [ 0.000815346367882, 0.0009301516787421183, 0.0013556873516122592,0.0018067990163216174,
0.002231661744244805,
0.002566623111061489,
0.002742371472470933,
0.0026934687025337205,
0.002367128705599091,
0.001733171993546934,
0.0007937619733521335,
-0.0004101169669870894,
-0.0017969140438467452,
-0.003247443662375688,
-0.0046127321322768705,
-0.005727405997694832,
-0.006428067623365728,
-0.006574385122289071,
-0.006070108943401453,
-0.004881697719649887,
-0.0030523207066668807,
-0.0007085899604062147,
0.001942360944649289,
0.00462643301740858,
0.0070260977623323026,
0.008812189592791104,
0.009681439457397379,
0.009396202996405754,
0.007821853947342845,
0.004957046237918892,
0.0009525995051717928,
-0.003884253819476725,
-0.009102266546950158,
-0.014135562906116846,
-0.01834837774216477,
-0.021091637241645485,
-0.021766378372696574,
-0.019887902208111232,
-0.01514393126576582,
-0.007440195764992871,
0.003072053956712789,
0.015990380809577702,
0.03067906576115481,
0.0463100572296567,
0.06192402912455657,
0.07650633044215188,
0.08907086537702237,
0.0987439653986767,
0.104840139641491,
0.10692224785026005,
0.104840139641491,
0.0987439653986767,
0.08907086537702237,
0.07650633044215188,
0.06192402912455657,
0.0463100572296567,
0.03067906576115481,
0.015990380809577702,
0.003072053956712789,
-0.007440195764992871,
-0.01514393126576582,
-0.019887902208111232,
-0.021766378372696574,
-0.021091637241645485,
-0.01834837774216477,
-0.014135562906116846,
-0.009102266546950158,
-0.003884253819476725,
0.0009525995051717928,
0.004957046237918892,
0.007821853947342845,
0.009396202996405754,
0.009681439457397379,
0.008812189592791104,
0.0070260977623323026,
0.00462643301740858,
0.001942360944649289,
-0.0007085899604062147,
-0.0030523207066668807,
-0.004881697719649887,
-0.006070108943401453,
-0.006574385122289071,
-0.006428067623365728,
-0.005727405997694832,
-0.0046127321322768705,
-0.003247443662375688,
-0.0017969140438467452,
-0.0004101169669870894,
0.0007937619733521335,
0.001733171993546934,
0.002367128705599091,
0.0026934687025337205,
0.002742371472470933,
0.002566623111061489,
0.002231661744244805,
0.0018067990163216174,
0.0013556873516122592,
0.0009301516787421183,  
0.000815346367882 ]

def plot_filter_response(fir_filter, sample_rate =44100, start_freq=10, end_freq=15e3):
  """
    Helper fun for showing a filter response.
  """
  start = time.time()
  fir_filter =cupy.array(fir_filter)

  moments = cupy.arange(1024 * 1024) / sample_rate 
  frequencies = [start_freq]
  while frequencies[-1] <= end_freq:
    frequencies.append(frequencies[-1] * 1.02)
  
  filter_responses = []
  for idx, freq in enumerate(frequencies):
    phase = moments * (2 * cupy.pi * freq)
    amplitudes = cupy.sin(phase)
    rms = cupy.sum(amplitudes** 2) / len(phase)
    filtered = cupyx.scipy.signal.oaconvolve(amplitudes, fir_filter, mode="same")
    filter_rms = cupy.sum(filtered ** 2) / len(phase)
    filter_responses.append(filter_rms / rms)

  # Get the actual results.
  filter_responses = [float(lazy_result) for lazy_result in filter_responses]
  filter_responses = 10 * np.log(filter_responses) / np.log(10) 
  print(f"Plotting took: {time.time() - start:.2f}s for {len(frequencies)} samples")
  plt.plot(frequencies, filter_responses)
  plt.grid()
  plt.show()

