import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift, rfft, irfft

def careful_fft(time, signal):
    '''
    Compute the FFT of `signal` and return the corresponding
    frequency range, with 0 centered, as well as the amplitudes.
    The frequency reported is ordinary (not angular) frequency,
    in the reciprocal of the time units. It is assumed that
    `time` is an array of evenly-spaced time samples.
    '''
    N = len(signal)
    dt = time[1]-time[0]
    
    return fftshift(fftfreq(N, dt)), fftshift(fft(signal))

def careful_rfft(time, signal):
    '''
    Compute the FFT of `signal` (real data) and return the corresponding
    frequency range, as well as the amplitudes. The frequency reported
    is ordinary (not angular) frequency, in the reciprocal of the time units.
    It is assumed that `time` is an array of evenly-spaced time samples.
    '''
    N = len(signal)
    dt = time[1]-time[0]
    
    return fftfreq(N, dt)[:N//2+1], rfft(signal)
