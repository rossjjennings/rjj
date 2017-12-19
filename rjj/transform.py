import numpy as np

def transform_points(matrix, *coords):
    '''
    Transform a series of points, given by a sequence of arrays,
    one for each coordinate, by a given matrix.
    
    Inputs
    ------
    matrix: The transformation matrix.
    coords: The coordinates of the points to be transformed 
            (x1, x2, x3, ...). For example, in two dimensions,
            an array of x-coordinates and an array of y-coordinates.
    
    Outputs
    -------
    Coordinates of the transformed points, in the same format
    as the original coordinates `coords`.
    '''
    points = np.stack(coords, axis=-1)
    return tuple(dim for dim in np.inner(matrix, points))

def qform_points(matrix, *coords):
    '''
    Evaluate a quadratic form at a series of points,
    given by a sequence of arrays, one for each coordinate.
    
    Inputs
    ------
    matrix: The matrix representing the quadratic form.
    coords: The coordinates of the points to be transformed 
            (x1, x2, x3, ...). For example, in two dimensions,
            an array of x-coordinates and an array of y-coordinates.
    
    Outputs
    -------
    Values of the quadratic form at the given points.
    '''
    points = np.stack(coords, axis=-1)
    transformed_points = np.inner(points, matrix)
    return np.sum(points*transformed_points, axis=-1)
