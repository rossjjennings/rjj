import sympy as sym

def symbol(name):
    '''
    Return an `expr` object representing a variable with the given name.
    '''
    return expr(sym.Symbol(name))

class expr:
    '''
    An object representing a symbolic expression which is compatible
    with (at least the more important built-in) numpy ufuncs.
    
    The main purpose of this is to allow sneaky reinterpretation of
    functions which perform numerical computations as symbolic, so that
    they can be, e.g., differentiated.
    
    Uses sympy under the hood. The sympy expression represented by
    an `expr` object `e` can be accessed as `e.sympy`.
    '''
    def __init__(self, sympy_expr):
        self.sympy = sympy_expr
    
    def __str__(self):
        return 'expr<{}>'.format(str(self.sympy))
    
    def __repr__(self):
        return repr(self.sympy)
    
    def __add__(self, other):
        if isinstance(other, expr):
            return expr(self.sympy + other.sympy)
        else:
            return expr(self.sympy + other)
    
    def __sub__(self, other):
        if isinstance(other, expr):
            return expr(self.sympy - other.sympy)
        else:
            return expr(self.sympy - other)
    
    def __mul__(self, other):
        if isinstance(other, expr):
            return expr(self.sympy * other.sympy)
        else:
            return expr(self.sympy * other)
    
    def __truediv__(self, other):
        if isinstance(other, expr):
            return expr(self.sympy / other.sympy)
        else:
            return expr(self.sympy / other)
    
    def __pow__(self, other):
        if isinstance(other, expr):
            return expr(self.sympy ** other.sympy)
        else:
            return expr(self.sympy ** other)
    
    def __neg__(self):
        return expr(-self.sympy)
    
    def __pos__(self):
        return expr(+self.sympy)
    
    def __abs__(self):
        return expr(abs(self.sympy))
    
    def __radd__(self, other):
        return expr(other + self.sympy)
    
    def __rsub__(self, other):
        return expr(other - self.sympy)
    
    def __rmul__(self, other):
        return expr(other * self.sympy)
    
    def __rtruediv__(self, other):
        return expr(other / self.sympy)
    
    def __rpow__(self, other):
        return expr(other ** self.sympy)
    
    def sin(self):
        return expr(sym.sin(self.sympy))
    
    def cos(self):
        return expr(sym.cos(self.sympy))
    
    def tan(self):
        return expr(sym.tan(self.sympy))
    
    def arcsin(self):
        return expr(sym.asin(self.sympy))
    
    def arccos(self):
        return expr(sym.acos(self.sympy))
    
    def arctan(self):
        return expr(sym.atan(self.sympy))
    
    def sinh(self):
        return expr(sym.sinh(self.sympy))
    
    def cosh(self):
        return expr(sym.cosh(self.sympy))
    
    def tanh(self):
        return expr(sym.tanh(self.sympy))
    
    def arcsinh(self):
        return expr(sym.asinh(self.sympy))
    
    def arccosh(self):
        return expr(sym.acosh(self.sympy))
    
    def arctanh(self):
        return expr(sym.atanh(self.sympy))
    
    def exp(self):
        return expr(sym.exp(self.sympy))
    
    def log(self):
        return expr(sym.log(self.sympy))
    
    def sqrt(self):
        return expr(sym.sqrt(self.sympy))
