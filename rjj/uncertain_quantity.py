import numpy as np
import astropy.units as u
import astropy.constants as c
from uncertainties import ufloat, umath

def uqty(value, uncertainty=0, unit=""):
    if hasattr(value, 'nominal_value') and hasattr(value, 'std_dev'):
        pass
    elif hasattr(value, 'value') and hasattr(value, 'uncertainty') and hasattr(value, 'unit'):
        uncertainty = value.uncertainty
        unit = value.unit
        value = ufloat(value.value, uncertainty)
    else:
        value = ufloat(value, uncertainty)
    return UncertainQuantity(value, unit)

def err(std_dev=1):
    return UncertainQuantity(ufloat(0, std_dev), "")


class UncertainQuantityMeta(type):
    """
    This metaclass saves a huge amount of redundant code by dynamically generating
    functions that will be called by numpy ufuncs.
    It makes assumptions about the instantiating class's __slots__, so it defines them.
    """
    def __new__(cls, name, bases, dct):
        dct['__slots__'] = ('value', 'unit')
        return super().__new__(cls, name, bases, dct)

    def __init__(cls, name, bases, dct):
        def angular_wrapper(name):
            def wrapped_operation(self):
                operate = getattr(umath, name)
                if self.unit.is_equivalent(u.Unit()):
                    result = operate(self.to(u.Unit()).value)
                elif self.unit.is_equivalent(u.Unit("rad")):
                    result = operate(self.to(u.Unit("rad")).value)
                else:
                    raise u.UnitTypeError(f"Cannot apply function '{name}' to quantity with unit '{str(self.unit)}'")
                return cls(result, u.Unit())
            return wrapped_operation

        def dimensionless_wrapper(name):
            def wrapped_operation(self):
                operate = getattr(umath, name)
                if self.unit.is_equivalent(u.Unit()):
                    result = operate(self.to(u.Unit()).value)
                else:
                    raise u.UnitTypeError(f"Cannot apply function '{name}' to quantity with unit '{str(self.unit)}'")
                return cls(result, u.Unit())
            return wrapped_operation

        circular_funcs = ['sin', 'cos', 'tan']
        identical_funcs = ['sinh', 'cosh', 'tanh', 'exp', 'expm1',
                           'log', 'log1p', 'log2', 'log10']
        renamed_funcs = [('arcsin', 'asin'), ('arccos', 'acos'), ('arctan', 'atan'),
                         ('arcsinh', 'asinh'), ('arccosh', 'acosh'), ('arctanh', 'atanh')]
        for func_name in circular_funcs:
            setattr(cls, func_name, angular_wrapper(func_name))
        for func_name in identical_funcs:
            setattr(cls, func_name, dimensionless_wrapper(func_name))
        for func_name, umath_name in renamed_funcs:
            setattr(cls, func_name, dimensionless_wrapper(umath_name))

class UncertainQuantity(metaclass=UncertainQuantityMeta):
    def __init__(self, value, unit):
        self.value = value
        self.unit = u.Unit(unit)

    def to(self, unit):
        conversion_factor = self.unit.to(unit)
        # must construct from ufloat to preserve covariance relationships
        return UncertainQuantity(self.value*conversion_factor, unit)

    def __str__(self):
        if self.unit == u.Unit():
            return f"{self.value}"
        else:
            return f"{self.value} {self.unit}"

    def __repr__(self):
        if self.unit == u.Unit():
            return f"{self.value!r}"
        else:
            return f"{self.value!r} {self.unit}"

    def __format__(self, spec):
        if self.unit == u.Unit():
            return f"{{:{spec}}}".format(self.value)
        else:
            return f"{{:{spec}}} {{!s}}".format(self.value, self.unit)

    def __add__(self, other):
        if hasattr(other, 'to') and callable(other.to):
            other_as_unit = other.to(self.unit)
            return UncertainQuantity(self.value + other_as_unit.value, self.unit)
        elif self.unit.is_equivalent(u.Unit()):
            conversion_factor = self.unit.to(u.Unit())
            return UncertainQuantity(self.value * conversion_factor + other, u.Unit())
        else:
            raise u.UnitConversionError(f"Cannot add dimensionless number to quantity with unit '{str(self.unit)}'")

    def __radd__(self, other):
        return self + other

    def __neg__(self, other):
        return UncertainQuantity(-self.value, self.unit)

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        if isinstance(other, u.UnitBase):
            return UncertainQuantity(self.value, self.unit*other)
        elif hasattr(other, 'unit') and isinstance(other.unit, u.UnitBase):
            return UncertainQuantity(self.value*other.value, self.unit*other.unit)
        else:
            return UncertainQuantity(self.value*other, self.unit)

    def __rmul__(self, other):
        return self * other

    def __pow__(self, other):
        return UncertainQuantity(self.value**other, self.unit**other)

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1

    def __eq__(self, other):
        try:
            equal = other.to(self.unit).value == self.value
        except u.UnitConversionError:
            return False
        else: return equal

    def __ne__(self, other):
        return not (self == other)

    def __lt__(self, other):
        return self.value.nominal_value*self.unit < other.value.nominal_value*other.unit

    def __le__(self, other):
        return self.value.nominal_value*self.unit <= other.value.nominal_value*other.unit

    def __gt__(self, other):
        return self.value.nominal_value*self.unit > other.value.nominal_value*other.unit

    def __ge__(self, other):
        return self.value.nominal_value*self.unit >= other.value.nominal_value*other.unit

    def sqrt(self):
        return UncertainQuantity(self.value**0.5, self.unit**0.5)
