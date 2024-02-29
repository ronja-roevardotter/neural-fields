import numpy as np
import collections.abc


"""
    kernels in original and fourier tranformation - resource: https://www.robots.ox.ac.uk/~az/lectures/ia/lect2.pdf"""

# # # - - - kernels in original domain - - - # # #

def gaussian(sigma, x):
    """
    x here is the distance-value ||x-y||ˆ2 = (x1-y1)ˆ2+(x2-y2)ˆ2 with (x1, x2)=(0, 0), i.e. distance to zero node
    i.e. the input is (y1**2+y2**2)
    """
    return (1/(2*np.pi*(sigma**2)))*np.exp(-(x/(2*(sigma**2))))

def gaussian_xy_input(sigma, x, y):
    """
    x, y here is the coordinate-values w.r.t. a zero point (0, 0)
    """
    return (1/(2*np.pi*(sigma**2)))*np.exp(-((x**2 + y**2)/(2*(sigma**2))))


# # # - - - kernels in Fourier domain - - - # # # EXPONENTOAL REQUIRES UPDATE ! ! !

#def f_gaussian(sigma, k, xi):
#    return np.exp(-(1/2) * (sigma**2) * (k**2 + xi**2) )

def f_gaussian(sigma, k):
    """
    inut k is given by wavenumber addition: k**2+xi**2
    """
    return np.exp((-2.0) * (np.pi**2) * (sigma**2) * k )



# # # - - - derivatives of kernels - - - # # #

def deriv_gaussian(sigma, x):
    """
    inut x is given by wavenumber addition: x**2+y**2
    whereas x,y are points of distance to zero-point
    """
    return -(np.sqrt(x)/(2*np.pi*(sigma**4)))*np.exp(-x/(2*(sigma**2)))

# # # - - - derivatives of f_kernels - - - # # # UPDATE REQUIRED ! ! !

    
def deriv_f_gaussian(sigma, k):
    """
    inut k is given by wavenumber addition: k**2+xi**2
    """
    return -np.sqrt(k) * 4 * np.pi**2 * sigma**2 * np.exp((-2) * np.pi**2 * k * sigma**2)