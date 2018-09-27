import numpy as np
from numpy import pi, sin, cos, exp, log, sqrt
from numpy.fft import fft, ifft, fftfreq, rfft, irfft, rfftfreq
from scipy.special import sinc
from scipy.optimize import brent
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
    phase = -2j*pi*shift*rfftfreq(a.shape[-1])
    return irfft(rfft(a)*np.exp(phase), len(a))

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
