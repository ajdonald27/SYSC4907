import numpy as np 
import matplotlib.pyplot as plt 

# constants 

# sample_rate = 4000 
# signal_frequency = 72 
# fft_sizes = [256, 512, 1024, 2048]

# amplitude = 1 
# sample_period = 1 / sample_rate 

# N = 1

#  #  generating time axis 
# time = np.arange(N) / sample_rate 
# # V(t) = Asin(2piFt); 
# signal = amplitude * np.sin(2 * np.pi * signal_frequency * time) 

# # numpy provides FFT function : fft_output = np.fft.rfft(signal)
# fft_output = np.fft.rfft(signal)
# #FFT output can have complex #s, convvert those to magnitudes : 
# magnitude = np.abs(fft_output) 

# # create frequency axis 
# frequencies = np.fft.rfftfreq(N, d=1 / sample_rate) 

# # frequencies[k] : frequencies represented in  bin K 

# # find strongest freq : 
# peak_index = np.argmax(magnitude)
# detected_frequency = frequencies[peak_index]

# # print FFT size, theoretical resolution, detected frequency, error from 72Hz (HB100) and collection duration time 
# frequency_res = sample_rate / N 
# collection_time = N / sample_rate 
# frequency_error = detected_frequency - signal_frequency 

# # plot each result for a given FFT size 

# plt.figure()
# plt.plot(frequencies, magnitude)
# plt.xlim(0, 200)
# plt.xlabel("Frequency (Hz)")
# plt.ylabel("Magnitude")
# plt.title(...)
# plt.grid(True)
# plt.show()



# # Loop structure 
# for N in fft_sizes: 
#     # generate time axis
#     sample_period = 1 / sample_rate 
#     # creating time values ; 
#     time = np.arange(N) / sample_rate 
#     # generate sine wave 
#     signal = amplitude * np.sin(2 * np.pi * signal_frequency * time )

#     # compute FFT 
#     fft_output = np.fft.rfft(signal)
#     # compute magnitude  
#     magnitude = np.abs(fft_output) 
#     # create frequency axis 
#     frequencies = np.fft.rfftfreq(N, d=1 / sample_rate)
#     # find strongest freq : 
#     peak_index = np.argmax(magnitude)
#     detected_frequency = frequencies[peak_index] 

#     # calculate resolution, duration and error 
#     freq_res = sample_rate / N 
#     collection_time = N / sample_rate 
#     error = detected_frequency - signal_frequency 

#     # print result 
#     print(f"For the following FFT size {N}:\n")
#     print(f"The Frequency resolution is {freq_res}\n")
#     print(f"The collected time is {collection_time}\n")
#     print(f"The Frequency error is {error}")

#     # plot spectrum 
#     plt.figure() 
#     plt.plot(frequencies, magnitude)
#     plt.xlim(0,200)
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.title('...')
#     plt.grid(True)
#     plt.show()

# Mini Project 2 code 

# define FFT of single size 
sample_rate = 4000 
signal_frequency = 72 
N = 1024 
ampltiude = 1.0 

# create time axis and sine wave (same as above)
time = np.arange(N) / sample_rate 

signal = ampltiude * np.sin(2 * np.pi * signal_frequency * time)

# FFT without a window (rectangle)

fft_no_window = np.fft.rfft(signal)
magnitude_no_window = np.abs(fft_no_window)

#create frequency axis 
freqs = np.fft.rfftfreq(N, d=1 / sample_rate)

# create hammming window 
hamming_window = np.hamming(N)

print(hamming_window[:5])
print(hamming_window[-5:])


# basic plotting 
plt.figure() 
plt.plot(time, hamming_window)
plt.xlabel('Time (s)')
plt.ylabel('Windowing coefficient')
plt.title('Hamming Window')
plt.grid()
plt.show() 

# apply the window 
windowed_signal = signal * hamming_window

plt.figure() 
plt.plot(time,signal, label='Original')
plt.plot(time,windowed_signal, label="Windowed")
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Original & Windowed Signals')
plt.legend() 
plt.grid()
plt.show() 

# FFT the windowed signal 
fft_hamming = np.fft.rfft(windowed_signal)
magnitude_hamming = np.abs(fft_hamming)

# normalize for visual comparison 

magnitude_no_window_normalized = (magnitude_no_window / np.max(magnitude_no_window))
magnitude_hamming_normalized = (magnitude_hamming / np.max(magnitude_hamming))

# plot seperate 

plt.figure() 
plt.plot(freqs, magnitude_no_window_normalized)
plt.xlim(0,200)
plt.xlabel(' Freq (Hz)')
plt.ylabel('Normalized Magnitude')
plt.title('FFT without window')
plt.grid() 
plt.show() 

# repeat for hamming windopw 

plt.figure() 
plt.plot(freqs, magnitude_hamming_normalized)
plt.xlim(0,200)
plt.xlabel(' Freq (Hz)')
plt.ylabel('Normalized Magnitude')
plt.title('FFT Hamming ')
plt.grid() 
plt.show() 

plt.figure()

plt.plot(
    freqs,
    magnitude_no_window_normalized,
    label="No window"
)

plt.plot(
    freqs,
    magnitude_hamming_normalized,
    label="Hamming"
)

plt.axvline(
    signal_frequency,
    linestyle="--",
    label="True frequency"
)

plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Normalized Magnitude")
plt.title("Effect of Hamming Window")
plt.legend()
plt.grid()
plt.show()

# use dB to make it easier to see the leakage consequences 
epsilon = 1e-12

db_no_window = 20 * np.log10(
    magnitude_no_window_normalized + epsilon
)

db_hamming = 20 * np.log10(
    magnitude_hamming_normalized + epsilon
)

plt.figure()

plt.plot(freqs, db_no_window, label="No window")
plt.plot(freqs, db_hamming, label="Hamming")

plt.axvline(
    signal_frequency,
    linestyle="--",
    label="True frequency"
)

plt.xlim(0, 200)
plt.ylim(-100, 5)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude (dB)")
plt.title("Spectral Leakage Comparison")
plt.legend()
plt.grid()
plt.show()