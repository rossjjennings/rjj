import numpy as np
from numpy.fft import fft, ifft, fftfreq, fftshift, ifftshift, rfft, irfft, rfftfreq

def fourier(time, signal):
    '''
    Compute the Fourier transform of `signal` and return the corresponding
    frequency range, with 0 centered, as well as the amplitudes.
    The frequency reported is ordinary (not angular) frequency,
    in the reciprocal of the time units. It is assumed that
    `time` is an array of evenly-spaced time samples.
    '''
    N = signal.shape[-1]
    dt = time[1]-time[0]
    freq = fftshift(fftfreq(N, dt))
    if 0 in time:
        signal = np.roll(signal, -time.tolist().index(0))
        spec = fftshift(fft(signal))*dt
    else:
        spec = fftshift(fft(signal))*dt
        spec *= np.exp(-2j*np.pi*freq*time[0])
    return freq, spec

def inv_fourier(freq, signal_fft):
    '''
    Compute the inverse Fourier transform of `signal_fft` and return a range of
    times, starting at zero, which are compatible with the given frequencies.
    This is the inverse of `fourier()`, and correspondingly accepts frequencies
    in the form output by that function (starting with the most negative).
    Returns a tuple of the form (time, signal).
    '''
    N = signal_fft.shape[-1]
    freq = ifftshift(freq)
    T = 1/(freq[1] - freq[0])
    time = np.linspace(0, T, N, endpoint=False)
    dt = time[1] - time[0]
    signal = ifft(signal_fft)/dt
    return time, signal

def real_fourier(time, signal):
    '''
    Compute the Fourier transform of `signal` (real-valued) and return the 
    corresponding frequency range, as well as the amplitudes.
    Only the positive-frequency portion of the Fourier transform is returned.
    The frequency reported is ordinary (not angular) frequency,
    in the reciprocal of the time units. It is assumed that
    `time` is an array of evenly-spaced time samples.
    '''
    N = signal.shape[-1]
    dt = time[1]-time[0]
    freq = rfftfreq(N, dt)
    if 0 in time:
        signal = np.roll(signal, -time.tolist().index(0))
        spec = rfft(signal)*dt
    else:
        spec = rfft(signal)*dt
        spec *= np.exp(-2j*np.pi*freq*time[0])
    return freq, spec

def inv_real_fourier(freq, signal_fft, odd=False):
    '''
    Compute the inverse Fourier transform of `signal_fft` and return a range of
    times, starting at zero, which are compatible with the given frequencies.
    This is the inverse of `real_fourier()`. Because calling `real_fourier()`
    on input of length `n` gives output of length `n//2+1`, the length of the
    original signal cannot be determined from the length of `signal_fft`.
    For this reason, two outputs are possible, one of even length and one of
    odd length. To get the odd-length output, set `odd=True`, otherwise the
    even-length output will be returned.
    Returns a tuple of the form (time, signal).
    '''
    N = 2*(signal_fft.shape[-1]-1) + (1 if odd else 0)
    T = 1/(freq[1] - freq[0])
    time = np.linspace(0, T, N, endpoint=False)
    dt = time[1] - time[0]
    signal = irfft(signal_fft, N)/dt
    return time, signal
