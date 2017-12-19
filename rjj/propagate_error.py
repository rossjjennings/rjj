import numpy as np
import sympy as sym
import inspect

def propagate_error(func, vals, errs):
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
    return func_val, func_err
