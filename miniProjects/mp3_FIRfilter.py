import numpy as np 
import matplotlib.pyplot as plt 
from scipy.signal import firwin, lfilter
# basics of FFT defines 
sample_rate = 4000 
desired_freq = 72 
interference_freq = 60 
noise_freq = 800 
duration = 1 

# amplitudes for all these 
desired_amp = 2.0 
inteference_amp = 0.75 
noies_amp = 0.5 


#creat time axis 
number_samples = int(sample_rate * duration)
time = np.arange(number_samples) / sample_rate 

# generate each componet as their on signal 

signal_72 = np.sin(2 * np.pi * desired_freq * time)
signal_noise = np.sin(2 * np.pi * noise_freq * time)
signal_inter = np.sin(2 * np.pi * interference_freq * time)

# combine them (superposition) 
raw_signal = signal_72 + signal_inter + signal_noise 


# plot a short portion of the time domain signal 
plt.figure() 
plt.plot(time,raw_signal)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.xlim(0,0.1)
plt.grid()
plt.show() 

# build a reusuable FFT function 

def computeFFT(signal, sample_rate): 
    N = len(signal)
    # fft the rfft (time -> freq -> time?)
    fft_output = np.fft.rfft(signal)
    magnitude = np.abs(fft_output)
    # normalize magnitude 
    magnitude = magnitude / np.max(magnitude)
    
    freqs = np.fft.rfftfreq(N, d=1 / sample_rate)
# frequencies = np.fft.rfftfreq(N, d=1 / sample_rate) 

    return freqs, magnitude


raw_freqs, raw_magnitude = computeFFT(raw_signal, sample_rate)

plt.figure()
plt.plot(raw_freqs, raw_magnitude)
plt.xlabel('Freq (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0,1000)
plt.title('output of computeFFT')
plt.grid()
plt.show()

# above : confirmed that we see all 3 peaks 

# Filter specs 
cutoff_freq = 300 
num_taps = 64 

coeffiecients = firwin(num_taps, cutoff_freq, fs=sample_rate)
print(coeffiecients)
print("number of coeffs:",len(coeffiecients))

#these coeffs are the h(k) values from the FIR equation
 
# apply the filter
filtered_signal = lfilter(coeffiecients, 1.0, raw_signal)

# 1.0 is the denominator coefficients, since it's an FIR. there's no feedback denominator beyond 1
 
# Filter startup delay : delay_samples = (num_taps - 1 ) / 2 
delay_samples = (num_taps - 1) / 2 
delay_seconds = (delay_samples) / sample_rate

# plot the time domain comparison 
#raw 
plt.figure() 
plt.plot(time, raw_signal)
plt.xlim(0,0.1)
plt.grid()
plt.show()

# filtered 

plt.figure() 
plt.plot(time, filtered_signal)
plt.xlim(0,0.1)
plt.show() 

# compute the filtered spectrum (take the FFT) 

filter_freq, filtered_magnitude = computeFFT(filtered_signal, sample_rate)

# plot it with an overlay
plt.figure() 
plt.plot(raw_freqs, raw_magnitude, label="before filtering")
plt.plot(filter_freq, filtered_magnitude, label="after filtering")

plt.xlim(0,1000)
plt.xlabel('Freq (Hz)')
plt.ylabel('Magnitude') 
plt.title('FIR Filter comparison') 
plt.legend() 
plt.grid() 
plt.show() 
# # EXAMPLE FUNCTION STRUCTURE import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import firwin, lfilter


# def generate_signal(...):
#     # Build 60 Hz, 72 Hz, and 800 Hz components
#     # Return time and combined signal


# def compute_spectrum(...):
#     # Compute rFFT
#     # Compute magnitudes
#     # Build frequency axis
#     # Return frequency and magnitude arrays


# def design_low_pass_filter(...):
#     # Call firwin
#     # Return coefficients


# def main():
#     # Define configuration

#     # Generate raw signal

#     # Verify raw spectrum

#     # Design FIR filter

#     # Apply FIR filter

#     # Compute filtered spectrum

#     # Plot results

#     # Print frequency measurements


# if __name__ == "__main__":
#     main()



# Make the attenuation easier to measure 

index_72 = np.argmin(np.abs(raw_freqs - 72))
index_60 = np.argmin(np.abs(raw_freqs - 60))
index_800 = np.argmin(np.abs(raw_freqs - 800))

print("800 Hz before : ", raw_magnitude[index_800])
print("800 Hz after filtering: ", filtered_magnitude[index_800]) 

# calculate the attenuation 
atten_db = 20 * np.log10(filtered_magnitude[index_800] / raw_magnitude[index_800])
print("Filter attenuation in dB is : ", atten_db)