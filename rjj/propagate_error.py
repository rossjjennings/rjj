import numpy as np
import sympy as sym
from collections import namedtuple
import inspect

value_with_error = namedtuple('value_with_error', ('value', 'error'))

def propagate_error(func, vals, errs):
    '''
    First-order Gaussian error propagation.
    
    Inputs
    ------
    func: A function taking any number of sympy Symbol objects
          and returning a scalar.
    vals: Floating-point numbers at which to evaluate the function.
    errs: Errors associated to the values `vals`.
    
    Outputs
    -------
    func_val: The value of the function, evaluated at the specified point.
    func_err: The error in the function value, determined by propagating
              the errors in the coordinates of the evaluation point.
    '''
    args = inspect.getargspec(func).args
    syms = {arg: sym.Symbol(arg) for arg in args}
    expr = func(**syms)
    grad = {arg: sym.diff(expr, syms[arg]) for arg in syms}
    
    func = sym.lambdify(tuple(syms.values()), expr, "numpy")
    for arg in grad:
        grad[arg] = sym.lambdify(tuple(syms.values()), grad[arg], "numpy")
    
    func_val = func(*vals)
    grad_vals = {arg: grad[arg](*vals) for arg in grad}
    func_err = np.sqrt(sum(grad_vals[arg]**2*err**2
                           for arg, err in zip(args, errs)))
    return value_with_error(func_val, func_err)
