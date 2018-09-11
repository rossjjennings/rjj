import numpy as np

def corr_2pt(arr1, arr2):
    '''
    A Python function that should be equivalent to np.correlate with mode='full'.
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

def intermediates(arr1, arr2):
    '''
    Return intermediate arrays which are summed to produce the autocorrelation.
    '''
    m, n = len(arr1), len(arr2)
    out = list()
    for k in range(m + n - 1):
        min2 = max(0, n - k - 1)
        max2 = min(m + n - k - 1, n)
        min1 = max(0, k - n + 1)
        max1 = min(k + 1, m)
        out.append(arr1[min1:max1] * arr2[min2:max2])
    return out

def bounds(arr1, arr2):
    m, n = len(arr1), len(arr2)
    out = list()
    for k in range(m + n - 1):
        min2 = max(0, n - k - 1)
        max2 = min(m + n - k - 1, n)
        min1 = max(0, k - n + 1)
        max1 = min(k + 1, m)
        out.append([(min1, max1), (min2, max2)])
    return out

def corr_3pt(arr1, arr2, arr3):
    m, n, p = len(arr1), len(arr2), len(arr3)
    inter = intermediates(arr1, arr2)
    rows = [corr_2pt(i, arr3) for i in inter]
    num_rows = m + n - 1
    max_length = min(m + n - 1, n + p - 1)
    first_max_index = min(m, n) - 1
    last_max_index = num_rows - min(m, n)
    
    out = np.ma.masked_all((num_rows, max_length))
    for i, row in enumerate(rows):
        start = max(0, i - last_max_index)
        out[i,start:start+len(row)] = row
    return out
