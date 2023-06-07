import numpy as np
import collections.abc


"""
    kernels in original and fourier tranformation - equations determined by WolframAlpha.
    originally according to publication 'Graph neural fields: 
    A framework for spatiotemporal dynamical models on the human connectome' 
    by Aqil, Atasoy, Kringelbach & Hindriks (2021), Table 2., but that was 'in the fourier domain' and not correct"""

# # # - - - kernels in original domain - - - # # #

def gaussian_distance(sigma, x):
    """
    x here is the distance-value ||x-y||ˆ2 = (x1-y1)ˆ2+(x2-y2)ˆ2 with (x1, x2)=(0, 0), i.e. distance to zero node
    """
    return (1/(2*np.pi*(sigma**2)))*np.exp(-(x/(2*(sigma**2))))

def gaussian(sigma, x, y):
    """
    x, y here is the coordinate-values w.r.t. a zero point (0, 0)
    """
    return (1/(2*np.pi*(sigma**2)))*np.exp(-((x**2 + y**2)/(2*(sigma**2))))


# # # - - - kernels in Fourier domain - - - # # # EXPONENTOAL REQUIRES UPDATE ! ! !

def f_gaussian(sigma, k, xi):
    return np.exp(-(1/2) * (sigma**2) * (k**2 + xi**2) )



# # # - - - derivatives of kernels - - - # # #

def deriv_gaussian(sigma, x):
    return -(x/(np.sqrt(2*np.pi)*sigma**3))*np.exp(-(((x**2))/(2*(sigma**2))))

# # # - - - derivatives of f_kernels - - - # # # UPDATE REQUIRED ! ! !

    
def deriv_f_gaussian(sigma, k):
    return -(sigma**2)*k*np.exp(-(1/2)*(sigma**2)*(k**2))