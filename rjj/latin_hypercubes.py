import random

def test_latin_hypercube(dim=3, n_samples=6):
    '''
    Tests for latin_hypercube()
    '''
    hypercube = latin_hypercube(dim, n_samples)
    
    # Check that the output has the right shape
    assert len(hypercube) == n_samples
    for i in range(n_samples):
        assert len(hypercube[i]) == dim
    
    # Check it's a latin hypercube
    for point in hypercube:
        for other in hypercube:
            if point != other:
                assert all(point[i] != other[i] for i in range(dim))

def test_symmetric_latin_hypercube(dim=3, n_samples=6):
    '''
    Tests for symmetric_latin_hypercube()
    '''
    hypercube = symmetric_latin_hypercube(dim, n_samples)
    
    # Check that the output has the right shape
    assert len(hypercube) == n_samples
    for i in range(n_samples):
        assert len(hypercube[i]) == dim
    
    # Check it's a latin hypercube
    for point in hypercube:
        for other in hypercube:
            if point != other:
                assert all(point[i] != other[i] for i in range(dim))
    
    # Check it's symmetric
    for point in hypercube:
        inverse = [n_samples - 1 - value for value in point]
        assert inverse in hypercube

def latin_hypercube(dim, n_samples):
    '''
    Generate a Latin hypercube.
    
    Inputs
    ------
    dim: The dimension of the hypercube to generate.
    n_samples: The number of samples per side.
    
    Outputs
    -------
    points: A list of points in the hypercube 
            (each a list of integers between 0 and dim-1).
    '''
    unused_values = [set(range(n_samples)) for i in range(dim)]
    points = []
    
    for i in range(n_samples):
        point = []
        
        for j in range(dim):
            value = random.choice(tuple(unused_values[j]))
            point.append(value)
            unused_values[j].remove(value)
        
        points.append(point)
    
    return points

def symmetric_latin_hypercube(dim, n_samples):
    '''
    Generate a symmetric Latin hypercube.
    
    Inputs
    ------
    dim: The dimension of the hypercube to generate.
    n_samples: The number of samples per side.
    
    Outputs
    -------
    points: A list of points in the hypercube 
            (each a list of integers between 0 and dim-1).
    '''
    if n_samples % 2 != 0:
        raise ValueError("Number of samples must be even.")
    
    unused_values = [set(range(n_samples)) for i in range(dim)]
    points = []
    
    for i in range(n_samples//2):
        point1 = []
        point2 = []
        
        for j in range(dim):
            value = random.choice(tuple(unused_values[j]))
            point1.append(value)
            unused_values[j].remove(value)
            
            inverse = n_samples - 1 - value
            point2.append(inverse)
            unused_values[j].remove(inverse)
        
        points.append(point1)
        points.append(point2)
    
    return points
