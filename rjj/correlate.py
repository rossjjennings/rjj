import numpy as np

def corr_2pt(arr1, arr2):
    '''
    A Python function that should be equivalent to np.correlate.
    '''
    m, n = len(arr1), len(arr2)
    out = list()
    for k in range(m + n - 1):
        min2 = max(0, n - k - 1)
        max2 = min(m + n - k - 1, n)
        min1 = max(0, k - n + 1)
        max1 = min(k + 1, m)
        out.append(np.sum(arr1[min1:max1] * arr2[min2:max2]))
    return np.array(out)

