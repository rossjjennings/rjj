from collections import Counter
from math import factorial

def partitions(n):
    '''
    Generate all partitions of the integer `n`.
    '''
    for p in boxed_partitions(n, n, n):
        yield p

def boxed_partitions(n, k, s):
    '''
    Generate all partitions of `n` into at most `k` parts, each of which is at most `s`.
    '''
    if n == 0:
        yield []
        return
    elif n > s*k:
        return
    elif n == s*k:
        p = [s]*k
        yield p
        return
    else:
        for m in range(1, min(n+1,s+1)):
            for p in boxed_partitions(n-m, k-1, m):
                yield [m] + p
        return

def decreasing_sequences(seq_max, seq_min, length):
    '''
    Generate all decreasing sequences of integers with the specified length and 
    maximum and minimum values `seq_max` and `seq_min`.
    '''
    if length == 0:
        yield []
        return
    else:
        for first in range(seq_min, seq_max + 1):
            for seq in decreasing_sequences(first, seq_min, length-1):
                yield [first] + seq
        return

def ndshk(n, s, k):
    '''
    Calculate the number of cases corresponding to each possible outcome when rolling
    `n` dice of `s` sides each and summing the largest `k` values that appear.
    Returns a dictionary mapping a sum to the number of cases for which it occurs.
    '''
    outcomes = {}
    for outcome in range(k, k*s + 1):
        cases = 0
        for partition in boxed_partitions(outcome - k, k, s - 1):
            pips = [1]*k
            for i, val in enumerate(partition):
                pips[i] += partition[i]
            for ignored in decreasing_sequences(min(pips), 1, n-k):
                pips_all = pips + ignored
                multiplicity = factorial(n)
                for val in Counter(pips_all).values():
                    multiplicity //= factorial(val)
                cases += multiplicity
        outcomes[outcome] = cases
    return outcomes
