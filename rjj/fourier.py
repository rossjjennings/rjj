import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft

def fourier(time, signal):
    '''
    Compute the Fourier transform of `signal` and return the corresponding
    frequency range, with 0 centered, as well as the amplitudes.
    The frequency reported is ordinary (not angular) frequency,
    in the reciprocal of the time units. It is assumed that
    `time` is an array of evenly-spaced time samples.
    '''
    N = len(signal)
    dt = time[1]-time[0]
    freq = fftshift(fftfreq(N, dt))
    if 0 in time:
        signal = np.roll(signal, -time.tolist().index(0))
        spec = fftshift(fft(signal))*dt
    else:
        spec = fftshift(fft(signal))*dt
        spec *= np.exp(-2j*np.pi*freq*time[0])
    return freq, spec

def real_fourier(time, signal):
    '''
    Compute the Fourier transform of `signal` (real data) and return the 
    corresponding frequency range, as well as the amplitudes.
    Only the positive-frequency portion of the Fourier transform is returned.
    The frequency reported is ordinary (not angular) frequency,
    in the reciprocal of the time units. It is assumed that
    `time` is an array of evenly-spaced time samples.
    '''
    N = len(signal)
    dt = time[1]-time[0]
    freq = fftfreq(N, dt)[:N//2+1]
    if 0 in time:
        signal = np.roll(signal, -time.tolist().index(0))
        spec = rfft(signal)*dt
    else:
        spec = rfft(signal)*dt
        spec *= np.exp(-2j*np.pi*freq*time[0])
    return freq, spec
