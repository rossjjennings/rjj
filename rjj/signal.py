import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from numpy.random import randn
from scipy.special import sinc
from scipy.optimize import brent, curve_fit
import warnings
import sys
eps = sys.float_info.epsilon

def dft_matrix(n, norm=None):
    '''
    The matrix representing the discrete Fourier transform on a cycle
    of length `n`. Calculating with the matrix directly is significantly
    slower than using the FFT.
    '''
    F = np.array([fft(row, norm=norm) for row in np.eye(n)])
    return F.T

def discrete_laplacian(n):
    '''
    The discrete Laplace operator for a cycle of length `n`.
    '''
    L = np.zeros((n, n))
    for i in range(n):
        L[i,i] = -2
        L[i,(i+1)%n] = 1
        L[i,(i-1)%n] = 1
    return L

def diagonalized_laplacian(n):
    '''
    A diagonal matrix conjugate to the discrete Laplace operator on a cycle
    of length `n`. This is equivalent to FLF^-1, where F is the DFT matrix,
    and is used in discrete_hermite().
    '''
    D = np.zeros((n, n))
    for k in range(n):
        D[k,k] = -4*sin(pi*k/n)**2
    return D

def discrete_hermite(n, m):
    '''
    Calculate the discrete Hermite function of length `n` and order `m`.
    This is the unit eigenvector corresponding to the (m+1)st largest 
    eigenvalue of the discrete Schrödinger Hamiltonian H = L + FLF^-1,
    where L is the discrete Laplacian and F is the DFT operator.
    Since H commutes with the DFT, this is also an eigenvector of the DFT.
    If m is `None`, return a matrix with all n eigenvectors as columns.
    '''
    H = -discrete_laplacian(n) - diagonalized_laplacian(n)
    eigvals, eigvecs = np.linalg.eigh(H)
    return eigvecs if m is None else eigvecs[:,m]

def fft_roll(a, shift):
    '''
    Roll array by a given (possibly fractional) amount, in bins.
    Works by multiplying the FFT of the input array by exp(-2j*pi*shift*f)
    and Fourier transforming back. The sign convention matches that of 
    numpy.roll() -- positive shift is toward the end of the array.
    This is the reverse of the convention used by pypulse.utils.fftshift().
    If the array has more than one axis, the last axis is shifted.
    '''
    n = a.shape[-1]
    warnings.filterwarnings(action='error', category=np.ComplexWarning)
    try:
        phase = -2j*pi*shift*rfftfreq(n)
        return irfft(rfft(a)*np.exp(phase), n)
    except np.ComplexWarning:
        phase = -2j*pi*shift*fftfreq(n)
        filtr = np.exp(phase)
        if n % 2 == 0:
            # Take real part of Nyquist frequency term
            # to match behavior when a is real
            filtr[n//2] = filtr[n//2].real
        return ifft(fft(a)*filtr, n)
    finally:
        warnings.resetwarnings()

def interp_ws(signal, ts = None):
    '''
    Calculate the Whittaker-Shannon interpolant of a signal.
    Returns a function computing the interpolant at a point `t`.
    `ts` is the array of sample points (assumed evenly-spaced).
    If `ts` is left unspecified, `arange(len(signal))` is used.
    '''
    if ts is None:
        ts = np.arange(len(signal))
    dt = ts[1] - ts[0]
    
    def interpolant(t):
        return np.sum(signal*sinc((t - ts)/dt))
    
    return interpolant

def eval_sin(t, amp, freq, phase, offset, cov=None):
    '''
    Evaluate a sine function with arbitrary amplitude, frequency,
    phase, and offset from zero. Primarily useful in conjunction with
    fit_sin() below. For example, to sample a sine curve fitted to
    a time series (t0, x0) at points t, can use 
    `eval_sin(t, **fit_sin(t0, x0))`.
    '''
    return amp * np.sin(2*pi*freq*t - phase) + offset

def fit_sin(t, x, return_cov=False, **kwargs):
    '''
    Fit a sine curve to a time series (t, x), using the frequency maximizing
    the FFT-based power spectrum as an initial guess for the frequency. 
    Returns the amplitude, frequency, phase, and offset of the optimal curve,
    along with the corresponding covariance matrix (if `return_cov` is `True`),
    in a dictionary, which can be used to sample it using eval_sin().
    '''
    freqs = np.fft.fftfreq(len(t), (t[1]-t[0]))   # assume uniform spacing
    fft = np.fft.fft(x)
    powspec = fft*np.conj(fft)
    guess_freq = freqs[1 + np.argmax(powspec[1:])]
    guess_amp = np.std(x) * np.sqrt(2)
    guess_offset = np.mean(x)
    guess_params = np.array([guess_amp, guess_freq, 0., guess_offset])

    (amp, freq, phase, offset), cov = curve_fit(eval_sin, t, x, p0=guess_params, **kwargs)
    params = {'amp': amp, 'freq': freq, 'phase': phase, 'offset': offset}
    if return_cov: params['cov'] = cov
    return params

def interp_ws(signal, ts = None):
    '''
    Calculate the Whittaker-Shannon (sinc) interpolant of a signal.
    Returns a function computing the interpolant at a point `t`.
    `ts` is the array of sample points (assumed evenly-spaced).
    If `ts` is left unspecified, `arange(len(signal))` is used.
    '''
    if ts is None:
        ts = np.arange(len(signal))
    dt = ts[1] - ts[0]
    
    def interpolant(t):
        return np.sum(signal*sinc((t - ts)/dt))
    
    return interpolant

def periodic_sinc(n, x):
    '''
    Calculate the "periodic sinc function": the Fourier transform of
    a windowed Dirac comb. Convolving this with a Nyquist sampled 
    periodic function gives back the original periodic function.
    The sample rate is `n` per period. This function should broadcast
    over array `x` values, and returns 0-dimensional arrays on scalars.
    '''
    if n % 2 == 0:
        return np.piecewise(x, [x % n == 0, x % n != 0], [1, lambda u: sin(pi*u)/(n*tan(pi*u/n))])
    else:
        return np.piecewise(x, [x % n == 0, x % n != 0], [1, lambda u: sin(pi*u)/(n*sin(pi*u/n))])

def interp_pws(signal, ts = None):
    '''
    Calculate the periodic Whittaker-Shannon (sinc) interpolant of a signal.
    Returns a function computing the interpolant at a point `t`.
    `ts` is the array of sample points (assumed evenly-spaced).
    If `ts` is left unspecified, `arange(len(signal))` is used.
    '''
    n = len(signal)
    if ts is None:
        ts = np.arange(n)
    dt = ts[1] - ts[0]
    
    def interpolant(t):
        return np.sum(signal*periodic_sinc(n, (t - ts)/dt))
    
    return interpolant

def rolling_sum(arr, size):
    '''
    Calculate the sum of values in `arr` in a sliding window of length `size`,
    wrapping around at the end of the array.
    '''
    n = len(arr)
    s = np.cumsum(arr)
    return np.array([s[(i+size)%n]-s[i]+(i+size)//n*s[-1] for i in range(n)])

def red_noise(index, size, pad_factor=2):
    '''
    Create a realization of power-law red noise with unit variance.
    Works around spectra that diverge at zero frequency by removing
    the mean of the generated time series.
    
    Inputs
    ------
    `index`: the slope of the power-law spectrum.
    `size`:  the number of samples to return.
    `pad_factor`: factor by which to over-generate samples so as to avoid 
                  periodic output.
    '''
    bufsize = pad_factor*size//2+1
    white_noise = randn(bufsize) + 1j*randn(bufsize)
    
    freq = rfftfreq(pad_factor*size)
    freq[0] = 1
    filtr = freq**(index/2)
    filtr[0] = 0
    filtr /= np.sqrt(np.mean(filtr**2))
    
    output_fft = filtr*white_noise
    output = irfft(output_fft, pad_factor*size)
    output /= np.sqrt(freq[1])
    
    return output[:size]

def powlaw_covmat(T, n, fmin, fmax, sigma2=1):
    '''
    Calculate the covariance matrix of power-law red noise with specified
    upper and lower cutoff frequencies. Uses a modified midpoint rule to
    compute the autocovariance function (as the Fourier transform of the
    power spectrum) for increased accuracy compared to the FFT.
    
    Inputs
    ------
    `T`: the duration of the time series.
    `n`: the number of sample points.
    `fmin`: lower cutoff frequency for the power-law spectrum.
    `fmax`: upper cutoff frequency for the power-law spectrum.
    `sigma2`: variance at lag 0.
    '''
    lags = np.linspace(0, T, n, endpoint=False)
    edges = np.linspace(fmin, fmax, 145)
    
    nodes = (3*(edges[1:]**-2 - edges[:-1]**-2))/(2*(edges[1:]**-3 - edges[:-1]**-3))
    weights = (edges[1:]**-3 - edges[:-1]**-3)/(edges[-1]**-3-edges[0]**-3)
    
    acf = sigma2*np.sum(weights*cos(2*pi*nodes*lags[:,np.newaxis]), axis=-1)
    
    indices = np.arange(n)
    xx, yy = np.meshgrid(indices, indices)
    covmat = acf[np.abs(xx-yy)]
    covmat += 1e-14*np.eye(n) # Fudge slightly to keep it positive definite
    
    return covmat
