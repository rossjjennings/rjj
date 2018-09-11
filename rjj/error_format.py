import numpy as np

def from_paren_err(qty):
    val, err = qty.strip(')').split('(')
    prec = len(val.split('.')[1])
    return float(val), int(err)*10**-prec

def to_paren_err(val, err, err_prec = 1):
    prec = err_prec - 1 - int(np.log10(err)//1)
    significant_digits = int(np.log10(val)//1) + prec + 1
    err_int = int(err*10**prec)
    format_string = '{{:.{}g}}({{}})'.format(significant_digits)
    return format_string.format(val, err_int)
