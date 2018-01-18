import numpy as np

def rotation_angle(covmat):
    '''
    The rotation angle (in radians) of the error ellipse associated to 
    the given covariance matrix. The rotation angle is always between
    -pi and pi, with an angle of 0 indicating that the major axis of the
    ellipse lies along the x-axis.
    '''
    x = covmat[0,0] - covmat[1,1]
    y = covmat[0,1] + covmat[1,0]
    return np.arctan2(y, x)/2

def axis_lengths(covmat, p = 0.05):
    '''
    The semimajor and semiminor axes of the error ellipse associated to
    the given covariance matrix. These are proportional to the square roots
    of the eigenvalues of the covariance matrix, but are calculated directly
    (rather than using an eigenvalue-finding algorithm).
    
    The scaling is chosen so that the probability of a bivariate normal
    random variable with the given covariance matrix falling outside the
    ellipse is exactly `p`. The default value of `p` is 0.05, which produces
    a 95% confidence region.
    '''
    scale_factor = np.sqrt(-2*np.log(p))
    tr = covmat[0,0] + covmat[1,1]
    det = covmat[0,0]*covmat[1,1] - covmat[0,1]*covmat[1,0]
    eigval1 = (tr + np.sqrt(tr**2 - 4*det))/2
    eigval2 = (tr - np.sqrt(tr**2 - 4*det))/2
    return scale_factor*np.sqrt(eigval1), scale_factor*np.sqrt(eigval2)
