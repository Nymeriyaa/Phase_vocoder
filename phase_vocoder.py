import numpy as np
from scipy.io import wavfile
import sys

def vocode (data: np.ndarray, stretch_factor : int, window_size: int , hop: int):
    phase_store = 0
    hanning_window = np.hanning(window_size)
    result = np.zeros (round(len(data) * stretch_factor + window_size))
        
    for i in np.arange(0, len(data)-(window_size + hop), round(hop / stretch_factor)):
        a1 = data[i: i + window_size]
        a2 = data[i + hop: i + window_size + hop]

        s1 =  np.fft.fft(hanning_window * a1)
        s2 =  np.fft.fft(hanning_window * a2)
        
        phase = np.angle(s2/s1)
        delta_phase = phase - hop * (window_size - 1) / 2 * 2*np.pi
        delta_phase = (delta_phase % 2*np.pi) - np.pi
        true_freq = (window_size - 1) / 2 * 2*np.pi + delta_phase / hop
        phase_store = phase_store + hop * true_freq
        
        a2_rephased = np.fft.ifft(np.abs(s2) * np.exp(1j*phase_store))
       
        i2 = int(i * stretch_factor)
        result[i2 : i2 + window_size ] += hanning_window * a2_rephased.astype(result[i2 : i2 + window_size].dtype)

    result = (result/result.max())
    return result.astype('float')

file_name = sys.argv[1]
file_out = sys.argv[2]
stretch_factor = float(sys.argv[3])
samplerate, wavdata = wavfile.read(file_name)
bit_depth = wavdata.max()
initial_data = wavdata.reshape(-1)
initial_data = np.array(initial_data, float)
initial_data /= bit_depth
frame_shift = 0.75 
window_size = 1024
hop = round((1 - frame_shift) * window_size)
data = np.copy(initial_data)
if initial_data.shape[0] % window_size != 0:
    data = np.concatenate((initial_data, np.zeros(window_size - initial_data.shape[0] % window_size)))

stretch_result =  vocode (data, stretch_factor, window_size, hop)
stretch_result *= bit_depth
stretch_result = np.asarray(stretch_result, int).astype('int16')
wavfile.write(file_out, samplerate, stretch_result)
