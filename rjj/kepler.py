import numpy as np
import numba

@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def kepler_tau(e, f):
    '''
    Calculate the normalized time, tau, since periapsis for a Keplerian orbit,
    given the orbital eccentricity, e, and true anomaly, f. Works for elliptic,
    parabolic, and hyperbolic orbits alike. The unit of time is sqrt(l^3/GM),
    where l is the semi-latus rectum.
    '''
    if e == 1:
        if f <= -np.pi or f >= np.pi:
            tau = np.nan
        else:
            tau = 1/2*(np.tan(f/2) + 1/3*np.tan(f/2)**3)
    elif 0 <= e < 1:
        tau = f/(1-e**2)**(3/2)
        arg = (((np.sqrt(1+e)-np.sqrt(1-e))*np.tan(f/2))
               /(np.sqrt(1+e)+np.sqrt(1-e)*np.tan(f/2)**2))
        tau -= 2/(1-e**2)**(3/2)*np.arctan(arg)
        tau -= (e*np.sin(f))/((1-e**2)*(1+e*np.cos(f)))
    elif e > 1:
        if f <= -np.arccos(-1/e) or f >= np.arccos(-1/e):
            tau = np.nan
        else:
            tau = (e*np.sin(f))/((e**2-1)*(1+e*np.cos(f)))
            arg = np.sqrt((e-1)/(e+1))*np.tan(f/2)
            tau -= 2/(e**2-1)**(3/2)*np.arctanh(arg)
    return tau

@numba.vectorize([numba.float64(numba.float64, numba.float64)])
def kepler_f(e, tau):
    '''
    Calculate the true anomaly, f, for a Keplerian orbit, given the orbital 
    eccentricity, e, and normalized time, tau, since periapsis, in units of
    sqrt(l^3/GM), where l is the semi-latus rectum. Works for elliptic,
    parabolic, and hyperbolic orbits alike.
    
    Works by finding a root of the equation kepler_tau(e, f) = tau using
    Newton's method, with a starting point selected by bisection. The result
    should be correct to within 2**-50 = 8.88e-16, or not much worse than
    machine precision.
    '''
    tol = 2**-50
    shift = 0
    if e < 1:
        period = 2*np.pi*(1-e**2)**(-3/2)
        shift = np.round(tau/period)
        tau -= shift*period
        fmax = np.pi
    elif e == 1:
        fmax = np.pi
    elif e > 1:
        fmax = np.arccos(-1/e)
    
    upper = fmax
    lower = -fmax
    while upper == fmax or lower == -fmax:
        midpt = (upper + lower)/2
        midpt_val = kepler_tau(midpt, e)
        if midpt_val > tau:
            upper = midpt
        elif midpt_val < tau:
            lower = midpt
        else:
            break
    
    prev_guess = upper
    guess = (upper + lower)/2
    while np.abs(guess - prev_guess) > tol:
        prev_guess = guess
        guess -= (kepler_tau(guess, e)-tau)*(1+e*cos(guess))**2
    
    return guess + 2*pi*shift
