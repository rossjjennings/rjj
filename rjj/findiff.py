import numpy as np
from scipy.special import factorial
eps = np.finfo(np.float64).eps

def findiff_coeffs(order=1, stencil_size=3, accuracy=1):
    '''
    Calculate finite difference coefficients.
    
    Inputs
    ------
    order: The order of the derivative to be calculated.
    stencil_size: The size of the stencil to be used.
    accuracy: The resulting estimate will be accurate to this order
              in the step size. Must be odd.
              
    Outputs
    -------
    stencil: The points of the stencil.
    coeffs: The finite difference coefficients corresponding to 
            the stencil points.
    '''
    x = np.linspace(-(stencil_size-1)/2, (stencil_size-1)/2, stencil_size)
    expts = np.arange(order % 2, order + accuracy, 2)
    A = np.stack([x**expt for expt in expts])
    c = np.array([(factorial(expt) if expt == order else 0) for expt in expts])
    return x, A.T @ np.linalg.inv(A @ A.T) @ c

def findiff(f, x, h, order=1, stencil_size=3, accuracy=1):
    '''
    Differentiate a function numerically.
    
    Inputs
    ------
    f: The function to be differentiated.
    x: Point(s) at which to calculate the derivative.
    h: The finite difference step size.
    order: The order of the derivative to be calculated.
    stencil_size: The size of the finite difference stencil to be used.
    accuracy: The resulting estimate will be accurate to this order
              in the step size. Must be odd.
              
    Outputs
    -------
    df: The derivative of f at the specified point(s).
    '''
    stencil, coeffs = findiff_coeffs(order, stencil_size, accuracy)
    return sum(coeff*f(x + point*h) for point, coeff in zip(stencil, coeffs))/h**order

def gridpt_diff(f, x, index, order=1, stencil_size=3, accuracy=1):
    '''
    Differentiate a function at a point numerically,
    given its values on an evenly-spaced grid.
    
    Inputs
    ------
    f: The function value at the grid points.
    x: The independent variable at the grid points.
    index: Index of the point at which to differentiate f.
    order: The order of the derivative to be calculated.
    stencil_size: The size of the finite difference stencil to be used.
                  Must be odd (so the stencil can be centered).
    accuracy: The resulting estimate will be accurate to this order
              in the step size. Must be odd.
              
    Outputs
    -------
    df: The derivative of f at the specified point.
    '''
    h = x[index+1]-x[index]
    stencil_width = (stencil_size-1)//2
    lower = index - stencil_width
    upper = index + stencil_width + 1
    stencil, coeffs = findiff_coeffs(order, stencil_size, accuracy)
    return np.dot(coeffs, f[...,lower:upper])/h**order
