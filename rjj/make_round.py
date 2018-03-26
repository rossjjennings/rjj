import numpy as np

def make_round(num):
    sign = np.sign(num)
    exponent = int(np.floor(np.log10(np.abs(num))))
    mantissa = np.abs(num/10**exponent)
    if mantissa < 2:
        round_mantissa = int(np.round(5*mantissa))/5
    elif mantissa < 5:
        round_mantissa = int(np.round(2*mantissa))/2
    else:
        round_mantissa = int(np.round(mantissa))/1
    return sign*round_mantissa*10**exponent
