import numpy as np

import py.kernels as ks

#import root and eigenvalue/-vector function
from scipy.optimize import root
from scipy.linalg import eigvals

from py.params import setParams

        
 # # # - - - # # # - - - - - - - - - - - - - - - - - - - # # # - - - # # #

   # #     TO DO : Check equations for voltage-based     # #
        
 # # # - - - # # # - - - - - - - - - - - - - - - - - - - # # # - - - # # #





# # # - - - # # # - - - transfer functions - - - # # # - - - # # #
    
def F_e(x, params):
    return 1/(1+np.exp(-params.beta_e*(x-params.mu_e)))


def F_i(x, params):
    return 1/(1+np.exp(-params.beta_i*(x-params.mu_i)))


# # # - - - # # # - - - derivatives of transfer functions - - - # # # - - - # # #

def derivF_e(x, params):
    return ((params.beta_e * np.exp(-params.beta_e*(x-params.mu_e)))/(np.exp(-params.beta_e*(x-params.mu_e))+1)**2)
    
def derivF_i(x, params):
    return ((params.beta_i * np.exp(-params.beta_i*(x-params.mu_i)))/(np.exp(-params.beta_i*(x-params.mu_i))+1)**2)


# # # - - - # # # - - - inverses of transfer functions - - - # # # - - - # # #

def inverseF_e(y, params):
    return params.mu_e - (1/params.beta_e) * np.log((1/y)-1)

def inverseF_i(y, params):
    return params.mu_i - (1/params.beta_i) * np.log((1/y)-1)


#### TODO: Vorzeichen an Gewichte anpassen, ich lenke die inhibition über w_ii und w_ei, nicht über die Formeln!!! #####

def activity_ui(ue, params):
    """Returns the excitatory nullcline w.r.t. ue (\frac{due}{dt}=0)
       for the activity-based model"""
    inside = params.w_ee * ue + params.I_e - inverseF_e(ue, params)
    return (1/params.w_ei) * inside

def activity_ue(ui, params):
    """Returns the inhibitory nullcline w.r.t. ui (\frac{ui}{dt}=0)
       for the activity-based model"""
    inside = inverseF_i(ui, params) + params.w_ii * ui - params.I_i
    return (1/params.w_ie) * inside


# # # - - - voltage-based nullclines - - - # # #

def voltage_ui(ue, params):
    """Returns the excitatory nullcline w.r.t. ui (\frac{ui}{dt}=0)
       for the voltage-based model"""
    return inverseF_i((1/params.w_ei) * (ue - params.w_ee*F_e(ue) - params.I_e))

def voltage_ue(ui, params):
    """Returns the, params inhibitory nullcline w.r.t. ui (\frac{ui}{dt}=0)
       for the voltage-based model"""
    return inverseF_e((1/params.w_ie) * (ui - params.w_ii*F_i(ui) - params.I_i))




#define linearization matrices

# # # - - - activity-based matrix - - - # # #
def activity_A11(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e
    return (1/params.tau_e) * (-1 + params.w_ee*derivF_e(Be, params))

def activity_A12(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e
    return (1/params.tau_e) * (-params.w_ei) * derivF_e(Be, params)

def activity_A21(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * params.w_ie * derivF_i(Bi, params)

def activity_A22(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * (-1 + (-params.w_ii)*derivF_i(Bi, params))

def activity_A(x, params):
    ue = x[0]
    ui = x[1]
    return [[activity_A11(ue, ui, params), activity_A12(ue, ui, params)], 
            [activity_A21(ue, ui, params), activity_A22(ue, ui, params)]]



# # # - - - voltage-based matrix - - - # # #
def voltage_A11(ue, params):
    return (1/params.tau_e) * (-1 + params.w_ee*derivF_e(ue))

def voltage_A12(ui, params):
    return (1/params.tau_e) * (-params.w_ei) * derivF_i(ui)

def voltage_A21(ue, params):
    return (1/params.tau_i) * params.w_ie * derivF_e(ue)

def voltage_A22(ui, params):
    return (1/params.tau_i) * (-1 + (-params.w_ii)*derivF_i(ui))

def voltage_A(x, params):
    ue = x[0]
    ui = x[1]
    return [[voltage_A11(ue, params), voltage_A12(ui, params)], 
            [voltage_A21(ue, params), voltage_A22(ui, params)]]


#define the activity-based model

def activity(x, params):
    ue = x[0]
    ui = x[1]
    
    exc_rhs = ((1/params.tau_e) * (-ue + F_e(params.w_ee*ue - params.w_ei*ui + params.I_e, params)))
    
    inh_rhs = ((1/params.tau_i) * (-ui + F_i(params.w_ie*ue - params.w_ii*ui + params.I_i, params)))
    
    return [exc_rhs, inh_rhs]



#define the voltage-based model

def voltage(x, params):
    ue = x[0]
    ui = x[1]
    
    exc_rhs = ((1/params.tau_e) * (-ue + params.w_ee * F_e(ue) - params.w_ei * F_i(ui) + params.I_e))
    
    inh_rhs = ((1/params.tau_i) * (-ui + params.w_ie * F_e(ue) - params.w_ii * F_i(ui) + params.I_i))
    
    return [exc_rhs, inh_rhs]


#function to determine the fixed points, depending on the model-type


def computeFPs(pDict):
    """ Derive all fixed points and collect them in the list fixed_points """
    
    params = setParams(pDict)
    fixed_points=[]
    
    if params.mtype == 'activity':
        start = 0
        end = 1
    else:
        start = -80
        end = 80

    for i in np.linspace(start, end, 61):
        if params.mtype == 'activity':
            sol = root(activity, [i, i], args=(params,), jac=activity_A, method='lm')#, method='lm')
          #  print('solution to root: ', sol.x)
        else:
          #  print('voltage(x): ', voltage([i,i]))
            sol = root(voltage, [i,i], args=(params,), jac=voltage_A, method='lm')
       # fix_point = [round(sol.x[0], 8), round(sol.x[1], 8)]
        if sol.success:
            if params.mtype == 'activity':
                closeness = all(np.isclose(activity(sol.x, params), [0.0, 0.0]))
            else:
                closeness = all(np.isclose(voltage(sol.x, params), [0.0, 0.0]))
            if closeness:
                if len(fixed_points)==0: #always append the firstly derived fixed point
                    fixed_points.append(sol.x)
                else:
                    already_derived = False
                    for k in range(len(fixed_points)):
                        if all(np.isclose(sol.x, fixed_points[k], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                            #fixed_points.append(sol.x)
                    if already_derived: #all(array):
                        pass #skip the already derived fixed points
                    else:
                        fixed_points.append(sol.x)
                        
    fixed_points = np.sort(fixed_points, axis=0)
    
    return fixed_points

#function to determine the stability of fixed points, depending on the fixed points AND model-type

                    
def checkFixPtsStability(fixed_points, params):
    
    # fixed_points = np.sort(fixed_points, axis=0)
    stability = []
    for i in range(len(fixed_points)):
        ue0 = fixed_points[i][0]
        ui0 = fixed_points[i][1]
        y=[ue0, ui0]
        if params.mtype == 'activity':
            A = activity_A(y, params)
        else:
            A = voltage_A(y, params)
        w = eigvals(A)
        if all(elem.real<0 for elem in w):
            stability.append(1)
        else: 
            stability.append(0)
    return stability


# # # # # # - - - - -                                               - - - - - # # # # # #
# # # # # # - - - - - Functions for Turing Stability Analysis below - - - - - # # # # # #
# # # # # # - - - - -                                               - - - - - # # # # # #


def f_kernel(sigma, k, k_string='gaussian'):
    
    kernel_func = getattr(ks, 'f_'+k_string)
    
    return kernel_func(sigma, k)

def deriv_f_kernel(sigma, k, k_string='gaussian'):
    
    kernel_func = getattr(ks, 'deriv_f_'+k_string)
    
    return kernel_func(sigma, k)


def a_jkValues(fp, params):
    
    exc = fp[0]
    inh = fp[1]
    
    if params.mtype=='activity':
        b_e = params.w_ee*exc - params.w_ei*inh + params.I_e
        b_i = params.w_ie*exc - params.w_ii*inh + params.I_i
        
        a_ee = params.w_ee * derivF_e(b_e, params)
        a_ei = params.w_ei * derivF_e(b_e, params)
        a_ie = params.w_ie * derivF_i(b_i, params)
        a_ii = params.w_ii * derivF_i(b_i, params)
        
    else:
        a_ee = params.w_ee * derivF_e(exc, params)
        a_ei = params.w_ei * derivF_i(inh, params)
        a_ie = params.w_ie * derivF_e(exc, params)
        a_ii = params.w_ii * derivF_i(inh, params)
        
    return a_ee, a_ei, a_ie, a_ii


        
                    
# # # - - - LINEARIZATION MATRIX - - - # # #
        
# # # - - - turing-linearization matrix - - - # # #
#Since we already disinguish the types by
#a_jk, the resulting linearization matrix is the same.

def turing_A11(k, a_ee, params):
    return (1/params.tau_e)*(-1 + a_ee*f_kernel(params.sigma_e, k, params.kernel))

def turing_A12(k, a_ei, params):
    return (1/params.tau_e)*(-a_ei)*f_kernel(params.sigma_i, k, params.kernel)

def turing_A21(k, a_ie, params):
    return (1/params.tau_i)*a_ie*f_kernel(params.sigma_e, k, params.kernel)

def turing_A22(k, a_ii, params):
    return (1/params.tau_i)*(-1 + (-a_ii)*f_kernel(params.sigma_i, k, params.kernel))

def turing_A(k):
    return [[turing_A11(k, a_ee, params), turing_A12(k, a_ei, params)],
            [turing_A21(k, a_ie, params), turing_A22(k, a_ii, params)]]


# # # - - - TRACE - - - # # #
    
def tr(k, a_ee, a_ii, params):
    return turing_A11(k, a_ee, params) + turing_A22(k, a_ii, params)

def dtr(k, a_ee, a_ii, params):
    return (1/params.tau_e)*a_ee*deriv_f_kernel(params.sigma_e, k, params.kernel)-(1/params.tau_i)*a_ii*deriv_f_kernel(params.sigma_i, k, params.kernel)
    
    
# # # - - - DETERMINANT - - - # # # 
#NOTE: calling the wrong fps (voltage vs activity) can turn the determinante upside done
def det(k, a_ee, a_ei, a_ie, a_ii, params):
    
    return (turing_A11(k, a_ee, params)*turing_A22(k, a_ii, params))-(turing_A12(k, a_ei, params)*turing_A21(k, a_ie, params))
    
    
    
# # # - - - the functions to test for turing instability - - - # # #

def pos_det(a_ee, a_ei, a_ie, a_ii, params):
    if det(0, a_ee, a_ei, a_ie, a_ii, params)>0:
        return True
    else: 
        return False
    
def neg_tr(k, a_ee, a_ii, params):
    if all(tr(k, a_ee, a_ii, params)<0):
        return True
    else:
        return False
    
def det_traj(k, a_ee, a_ei, a_ie, a_ii, params):
    if any(det(k, a_ee, a_ei, a_ie, a_ii, params)< -0.1**16):
        return True
    else:
        return False
    
def lmbd(k_real, a_ee, a_ei, a_ie, a_ii, params):
    k = k_real.astype(complex)
    lmbd_plus = (1/2)*(tr(k, a_ee, a_ii, params) + np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    lmbd_minus = (1/2)*(tr(k, a_ee, a_ii, params) - np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    return [lmbd_plus, lmbd_minus]



    
# # # - - - the functions to check Turing instability and check possibility of spatiotemporal patterns - - - # # #

def Turing_Hopf(k, a_ee, a_ii, a_ei, a_ie, params):
    for k_in in k:
        sol1 = root(tr, k_in, args=(a_ee, a_ii, params), method='hybr')
        sol2 = root(dtr, k_in, args=(a_ee, a_ii, params), method='hybr')
    if sol1.success:
        trace_root = True
    else:
        trace_root = False
    if sol2.success:
        deriv_trace = True
    else:
        deriv_trace = False
        
    determ = pos_det(a_ee, a_ei, a_ie, a_ii, params)
    
    return trace_root, deriv_trace, determ
        