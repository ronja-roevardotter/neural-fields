import numpy as np

import py.kernels as ks

#import root and eigenvalue/-vector function
from scipy.optimize import root
from scipy.linalg import eigvals

from py.params import setParams

from py.analysis import F_e, F_i, F_a, derivF_e, derivF_i, derivF_a
from py.analysis import a_jkValues, f_kernel
from py.funcs import getSwitchArray, getSwitchIndex, getCommonElement


# # # - - - # # # - - - adaptation matrix A(k)\in\mathbb{C}^{3\times3} - - - # # # - - - # # #

def a11(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_e) * (-1 + a_ee*f_kernel(params.sigma_e, k, params.kernel))

def a12(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_e) * (-a_ei) * f_kernel(params.sigma_i, k, params.kernel)

def a13(fp, params):
    ue = fp[0]
    ui = fp[1]
    be = params.w_ee * ue - params.w_ei * ui - params.b * derivF_a(ue, params) + params.I_e
    
    return (params.b/params.tau_e) * derivF_e(be, params)

def a21(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    return (1/params.tau_i) * a_ie * f_kernel(params.sigma_e, k, params.kernel)

def a22(k, fp, params):
    ue = fp[0]
    ui = fp[1]
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)

    return (1/params.tau_i)*(-1 + (-a_ii) * f_kernel(params.sigma_i, k, params.kernel))

def a23():
    return 0

def a31(fp, params):
    ue = fp[0]
    ui = fp[1]
    return (1/params.tau_a) * derivF_a(ue, params)

def a32():
    return 0

def a33(params):
    return -(1/params.tau_a)


# # # - - - # # # - - - polynomial entries - - - # # # - - - # # #


def c0(k, fp, params):
    return (-a11(k, fp, params) * a22(k, fp, params) * a33(params) + 
            a22(k, fp, params) * a13(fp, params) * a31(fp, params) + 
            a33(params) * a12(k, fp, params) * a21(k, fp, params))


def c1(k, fp, params):
    return (a11(k, fp, params) * a22(k, fp, params) + 
            a11(k, fp, params) * a33(params) + 
            a22(k, fp, params) * a33(params) - 
            a13(fp, params) * a31(fp, params) - 
            a12(k, fp, params) * a21(k, fp, params))

def c2(k, fp, params):
    return -a11(k, fp, params)-a22(k, fp, params)-a33(params)


# # # - - - # # # - - - conditions - - - # # # - - - # # #


def checkZeroEigval(k, fp, params):
    if all(c0(k, fp, params) < 0) or all(c0(k, fp, params) > 0):
        return 0
    else:
        return 1
    

def checkImagEigval(k, fp, params):
    temp = c1(k, fp, params) * c2(k, fp, params) - c0(k, fp, params)
    
    if all(temp < 0) or all(temp > 0):
        return 0
    else:
        indeces = getSwitchIndex(temp)
        c1_sign = c1(k[indeces], fp, params)
        if any(c1_sign >= 0):
            return 1
        else:
            return 0
    

def checkTakensBogdanov(k, fp, params):
    c0_array = c0(k, fp, params)
    c1_array = c1(k, fp, params)
    if (all(c0_array < 0) or all(c0_array > 0)) and (all(c1_array < 0) or all(c1_array > 0)):
        return 0
    else:
        c0_indeces = getSwitchIndex(c0_array)
        c1_indeces = getSwitchIndex(c1_array)
        same_k0 = getCommonElement(c0_indeces, c1_indeces)
        if same_k0:
            return 1
        else:
            return 0
        


# # # - - - # # # - - - conditions - - - # # # - - - # # #


def checkStability(k, fp, params):
    zeroVal = checkZeroEigval(k, fp, params)
    imagVal = checkImagEigval(k, fp, params)
    if not imagVal:
        doubleVal = checkTakensBogdanov(k, fp, params)
    else:
        doubleVal = 0
    
    return zeroVal, imagVal, doubleVal
    
    
    

    













