import numpy as np
import collections.abc


"""
    kernels in original and fourier tranformation - equations determined by WolframAlpha.
    originally according to publication 'Graph neural fields: 
    A framework for spatiotemporal dynamical models on the human connectome' 
    by Aqil, Atasoy, Kringelbach & Hindriks (2021), Table 2., but that was 'in the fourier domain' and not correct"""

# # # - - - kernels in original domain - - - # # #

def exponential(sigma, x):
    return (1/(2*sigma))*np.exp(-np.abs(x)/sigma) #np.sqrt(x*x))

def gaussian(sigma, x):
    return (1/2*(sigma))*np.exp(-0.5*((x**2)/((sigma**2)*np.pi)))  #1/(np.pi*(sigma**2))*np.exp(-0.5*(x/sigma)**2)


#Let's call it Wizard-Hat instead of Mexican-hat for multiple good reasons.
def wizard_hat(sigma, x):
    return (1-(x/sigma)**2)*np.exp(-(x*x)/(2*(sigma**2)))


# # # - - - kernels in Fourier domain - - - # # #

def f_exponential(sigma, k):
    return (1/(np.sqrt(2*np.pi)))*(1/(1+((sigma**2)*(k**2))))#scale*(1/((alpha**2) + (k**2)))

def f_gaussian(sigma, k):
    return  1/(2)*np.exp(-0.5*((k**2)*(sigma**2)))#1/(np.pi*(sigma**2))*sigma*np.exp(-((sigma*k)**2)/2)

def f_wizard_hat(sigma, k):
    #This Fourier transformation is still from the graph neural fields publication, I need to determine it another time.
    return (k**2)*np.exp(-((sigma*k)**2)/2)


# # # - - - derivatives of kernels - - - # # #


# # # - - - derivatives of f_kernels - - - # # #

    
def deriv_f_gaussian(sigma, k):
    return -(sigma**3)*k*np.exp(-(1/2)*(sigma**2)*(k**2))

def deriv_f_exponential(sigma, k):
    scale = 1/(2*sigma)
    alpha = 1/sigma
    #determined with wolfram-alpha
    return -((2*scale*sigma)/(sigma*k**2+alpha)**2)