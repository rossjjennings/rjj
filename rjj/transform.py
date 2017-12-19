import numpy as np

def transform_points(matrix, *coords):
    points = np.stack(coords, axis=-1)
    return tuple(dim for dim in np.inner(matrix, points))

def qform_points(matrix, *coords):
    points = np.stack(coords, axis=-1)
    transformed_points = np.inner(points, matrix)
    return np.sum(points*transformed_points, axis=-1)
