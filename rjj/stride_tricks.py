import numpy as np

def diag_blocks(arr):
    """
    Return the contents of the central diamond-shape region of a square matrix.
    Rows of the output correspond to diagonals of the original matrix, read from
    upper left to lower right, with the first row corresponding to the uppermost
    diagonal with the appropriate length.
    Two matrices are returned. The first, "wide" block, contains entries from
    diagonals where two entries fall on the bounding diamond, and the second,
    "narrow" block, contains entries from the remaining diagonals.
    Entries from the main diagonal are in the wide block when the size of the
    matrix is congruent to 1 or 2 modulo 4, and in the narrow block otherwise.
    For an n×n matrix, the shape of the wide block is (n//2, n//2+1) if n is even,
    and ((n+1)//2, (n+1)//2) if n is odd, while the shape of the narrow block is
    (n//2+1, n//2) if n is even, and ((n-1)//2, (n-1)//2) if n is odd.
    """
    m, n = arr.shape
    assert m == n
    s = arr.dtype.itemsize
    if n % 2 == 0:
        wide = np.lib.stride_tricks.as_strided(
            arr.flat[n//2-1:],
            shape=(n//2, n//2+1),
            strides=((n-1)*s, (n+1)*s),
        )
        narrow = np.lib.stride_tricks.as_strided(
            arr.flat[n//2:],
            shape=(n//2+1, n//2),
            strides=((n-1)*s, (n+1)*s),
        )
    else:
        wide = np.lib.stride_tricks.as_strided(
            arr.flat[(n-1)//2:],
            shape=((n+1)//2, (n+1)//2),
            strides=((n-1)*s, (n+1)*s),
        )
        narrow = np.lib.stride_tricks.as_strided(
            arr.flat[(3*n-1)//2:],
            shape=((n-1)//2, (n-1)//2),
            strides=((n-1)*s, (n+1)*s),
        )
    return wide, narrow
