import numpy as np
from scipy.linalg import svd

def concatenateReal(us, vs):
    """
    Preprocess the x- & y components of all VVFs such that the variables (i.e. positions (x,y)) are in columns, the observations (i.e. time stamps) in the rows.
    Out of two options, this option concatenates  rearranged utilde & vtilde into one real-valued matrix of dimension (T x 2(XxY)) w_real = [utilde|vtilde]

    INPUT:
    :us, vs: time series of x- and y-component vvfs (T x X x Y)-dimensional, array of arrays

    OUTPUT: 
    :w_real: (T x 2(XxY))-dimensional matrix of concatenated, rearranged us & vs: [utilde|vtilde]
    
    """

    T = us.shape[0]
    X = us.shape[1]
    Y = us.shape[2]

    us_flat = np.zeros((T, X*Y ))
    vs_flat = np.zeros((T, X*Y ))

    #rearrange (combine) spatial dimensions x & y in on position (i.e. (x,y)-coordinates are counted and flattened)
    for t in range(T):
        us_flat[t] = us[t].flatten()
        vs_flat[t] = vs[t].flatten()
    
    w_real = np.concatenate((us_flat, vs_flat), axis=1)

    return w_real 


def concatenateComplex(us, vs):
    """
    Preprocess the x- & y components of all VVFs such that the variables (i.e. positions (x,y)) are in columns, the observations (i.e. time stamps) in the rows.
    Out of two options, this option concatenates  rearranged utilde & vtilde into one complex-valued matrix of dimension (T x (XxY)) w_complex = utilde + i*vtilde

    INPUT:
    :us, vs: time series of x- and y-component vvfs (T x X x Y)-dimensional, array of arrays

    OUTPUT: 
    :w_complex: (T x (XxY))-dimensional matrix of concatenated, rearranged us & vs: utilde + i*vtilde

    """

    T = us.shape[0]
    X = us.shape[1]
    Y = us.shape[2]

    #rearrange (combine) spatial dimensions x & y in on position (i.e. (x,y)-coordinates are counted and flattened) 
    us_flat = np.zeros((T, X*Y ))
    vs_flat = np.zeros((T, X*Y ))

    for t in range(T):
        us_flat[t] = us[t].flatten()
        vs_flat[t] = vs[t].flatten()

    w_complex = us_flat + 1j*vs_flat

    return w_complex


def singularVectorDecomposition(w, all=False):
    """
    Determines the singular vector decomposition of the (T x N)-dimensional array w, either complex or real.

    INPUT:
    :w: (T x N)-dimensional array w, either complex or real
    :all: boolean to either get all spatial modes (True) or just as many as time steps (False, faster)

    OUTPUT: 
    :spatialModes: (N x T)-dimensional array of arrays, describing the kth spatial mode in column k
    :temporalModes: (T x T)-dimensional array of arrays, describing the time course in row k of the kth spatial model
    :singularValues: singular values of w, allowing computations of proportion of the overall variance given per kth spatial mode

    """

    #use off-shelf implementation for SVD
    #Note 1: returns the transpose of the spatialModes-matrix
    #Note 2: use full_matrices=False to acquire compact form of both Modes-matrices (since N is potentially very large and I don't need that many)
    temporalModes, singularValues, spatialModes_transpose = svd(w, full_matrices=all)

    return spatialModes_transpose.T, temporalModes, singularValues


def proportionOfVariance(Sigma):
    """
    Determine the proportion of the overall variance explained per spatial mode by Sigma[k]/sum(Sigma)
    
    INPUT:
    :Sigma: T-dimensional array of singular values explaining their share of contribution to the pattern

    OUTPUT:
    :variances: T-dimensional array where each entry explains the variance of that spatial mode
    """

    variances = np.zeros(len(Sigma))
    sums = np.sum(Sigma)

    for idx, val in enumerate(Sigma):
        variances[idx] = val/sums

    return variances


def reshapeSpatialModes(R, u_original):
    """
    Reshapes the (N x T)-dimensional array of arrays that consists of the spatial modes determined by the SVD of w (either real or complex).

    INPUT:
    :R: (N x T)-dim array of arrays whose kth column describes the kth spatial mode
    :u_original:  (T x X x Y) dimensional array of arrays that we need to identify the original dimensions for the reshaping

    OUTPUT:
    :xModes, yModes: (n x m x T) dimensional array where each (n x m)-grid corresponds to one mode for the x- and y-components, respectively (if R is real)
    or
    :spaceModes: (n x m x T) dimensional array where each (n x m)-grid corresponds to one mode, x-, or y-components are simultaneously considered due to the R complex
    """

    N = u_original.shape[1]
    M = u_original.shape[2]

    if np.all(np.isreal(R)):
        xModes = np.zeros((len(R), N, M))
        yModes = np.zeros((len(R), N, M))

        for idx, column in enumerate(R):
            #print(column)
            new = np.reshape(np.array(column), (2, N*M))
            xModes[idx] = np.reshape(new[0], (N,M))
            yModes[idx] = np.reshape(new[1], (N,M))
        
        return xModes, yModes
    else:
        spaceModes = np.reshape(R, (len(R), N, M))

        return spaceModes

        

