import numpy as np 
import matplotlib.pyplot as plt 

# constants 

sample_rate = 4000 
signal_frequency = 72 
fft_sizes = [256, 512, 1024, 2048]

amplitude = 1 
sample_period = 1 / sample_rate 

N = 1

 #  generating time axis 
time = np.arange(N) / sample_rate 
# V(t) = Asin(2piFt); 
signal = amplitude * np.sin(2 * np.pi * signal_frequency * time) 

# numpy provides FFT function : fft_output = np.fft.rfft(signal)
fft_output = np.fft.rfft(signal)
#FFT output can have complex #s, convvert those to magnitudes : 
magnitude = np.abs(fft_output) 

# create frequency axis 
frequencies = np.fft.rfftfreq(N, d=1 / sample_rate) 

# frequencies[k] : frequencies represented in  bin K 

# find strongest freq : 
peak_index = np.argmax(magnitude)
detected_frequency = frequencies[peak_index]

# print FFT size, theoretical resolution, detected frequency, error from 72Hz (HB100) and collection duration time 
frequency_res = sample_rate / N 
collection_time = N / sample_rate 
frequency_error = detected_frequency - signal_frequency 

# plot each result for a given FFT size 

plt.figure()
plt.plot(frequencies, magnitude)
plt.xlim(0, 200)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.title(...)
plt.grid(True)
plt.show()



# Loop structure 
for N in fft_sizes: 
    # generate time axis
    sample_period = 1 / sample_rate 
    # creating time values ; 
    time = np.arange(N) / sample_rate 
    # generate sine wave 
    signal = amplitude * np.sin(2 * np.pi * signal_frequency * time )

    # compute FFT 
    fft_output = np.fft.rfft(signal)
    # compute magnitude  
    magnitude = np.abs(fft_output) 
    # create frequency axis 
    frequencies = np.fft.rfftfreq(N, d=1 / sample_rate)
    # find strongest freq : 
    peak_index = np.argmax(magnitude)
    detected_frequency = frequencies[peak_index] 

    # calculate resolution, duration and error 
    freq_res = sample_rate / N 
    collection_time = N / sample_rate 
    error = detected_frequency - signal_frequency 

    # print result 
    print(f"For the following FFT size {N}:\n")
    print(f"The Frequency resolution is {freq_res}\n")
    print(f"The collected time is {collection_time}\n")
    print(f"The Frequency error is {error}")

    # plot spectrum 
    plt.figure() 
    plt.plot(frequencies, magnitude)
    plt.xlim(0,200)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('...')
    plt.grid(True)
    plt.show()

