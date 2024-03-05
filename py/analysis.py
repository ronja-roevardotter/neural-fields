import numpy as np

import sys
sys.path.append('/Users/ronja/Documents/GitHub/neural-fields/py')

import kernels as ks

#import root and eigenvalue/-vector function
from scipy.optimize import root
from scipy.linalg import eigvals

from params import setParams

        
 # # # - - - # # # - - - - - - - - - - - - - - - - - - - # # # - - - # # #

   # #     TO DO : Check equations for voltage-based     # #
        
 # # # - - - # # # - - - - - - - - - - - - - - - - - - - # # # - - - # # #





# # # - - - # # # - - - transfer functions - - - # # # - - - # # #
    
def F_e(x, params):
    return 1/(1+np.exp(-params.beta_e*(x-params.mu_e)))


def F_i(x, params):
    return 1/(1+np.exp(-params.beta_i*(x-params.mu_i)))

def F_a(x, params):
    return 1/(1+np.exp(-params.beta_a*(x-params.mu_a)))


# # # - - - # # # - - - derivatives of transfer functions - - - # # # - - - # # #

def derivF_e(x, params):
    return ((params.beta_e * np.exp(-params.beta_e*(x-params.mu_e)))/(np.exp(-params.beta_e*(x-params.mu_e))+1)**2)
    
def derivF_i(x, params):
    return ((params.beta_i * np.exp(-params.beta_i*(x-params.mu_i)))/(np.exp(-params.beta_i*(x-params.mu_i))+1)**2)

def derivF_a(x, params):
    return ((params.beta_a * np.exp(-params.beta_a*(x-params.mu_a)))/(np.exp(-params.beta_a*(x-params.mu_a))+1)**2)


# # # - - - # # # - - - inverses of transfer functions - - - # # # - - - # # #

def inverseF_e(y, params):
    return params.mu_e - (1/params.beta_e) * np.log((1/y)-1)

def inverseF_i(y, params):
    return params.mu_i - (1/params.beta_i) * np.log((1/y)-1)

def inverseF_a(y, params):
    return params.mu_a - (1/params.beta_a) * np.log((1/y)-1)


#### TODO: Vorzeichen an Gewichte anpassen, ich lenke die inhibition über w_ii und w_ei, nicht über die Formeln!!! #####

def activity_ui(ue, params):
    """Returns the excitatory nullcline w.r.t. ue (\frac{due}{dt}=0)
       for the activity-based model"""
    inside = params.w_ee * ue + params.I_e - inverseF_e(ue, params) - params.b*F_a(ue, params)
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
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e - params.b * F_a(ue, params)
    return (1/params.tau_e) * (-1 + derivF_e(Be, params) * (params.w_ee - params.b * derivF_a(ue, params))) 

def activity_A12(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui + params.I_e - params.b * F_a(ue, params)
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
    
    exc_rhs = ((1/params.tau_e) * (-ue + F_e(params.w_ee*ue - params.w_ei*ui - params.b * F_a(ue, params) + params.I_e, params)))
    
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
        if params.mtype == 'activity' and params.b == 0:
            A = activity_A(y, params)
        elif params.mtype == 'activity' and params.b != 0:
            A = adap_A(y, params)
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
    
    kernel_func = getattr(ks, 'deriv_f_' + k_string)
    
    return kernel_func(sigma, k)


def a_jkValues(fp, params):
    
    exc = fp[0]
    inh = fp[1]
    
    if params.mtype=='activity':
        b_e = params.w_ee*exc - params.w_ei*inh - params.b*F_a(exc, params) + params.I_e
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


# # # - - - LINEARIZATION MATRIX WITH ADAPTATIOn - from 2x2 to 3x3 - - - # # #

def adap_A11(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (-1 + params.w_ee*derivF_e(Be, params))

def adap_A12(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (-params.w_ei) * derivF_e(Be, params)

def adap_A13(ue, ui, params):
    Be = params.w_ee * ue - params.w_ei * ui - params.b * F_a(ue, params) + params.I_e
    return (1/params.tau_e) * (- params.b * derivF_e(Be, params))

def adap_A21(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * params.w_ie * derivF_i(Bi, params)

def adap_A22(ue, ui, params):
    Bi = params.w_ie * ue - params.w_ii * ui + params.I_i
    return (1/params.tau_i) * (-1 + (-params.w_ii)*derivF_i(Bi, params))

def adap_A23(ue, ui, params):
    return 0

def adap_A31(ue, ui, params):
    return (1/params.tau_a) * derivF_a(ue, params)

def adap_A32(ue, ui, params):
    return 0

def adap_A33(ue, ui, params):
    return -(1/params.tau_a)

def adap_A(x, params):
    ue = x[0]
    ui = x[1]
    return [[adap_A11(ue, ui, params), adap_A12(ue, ui, params), adap_A13(ue, ui, params)], 
            [adap_A21(ue, ui, params), adap_A22(ue, ui, params), adap_A23(ue, ui, params)], 
            [adap_A31(ue, ui, params), adap_A32(ue, ui, params), adap_A33(ue, ui, params)]]

        
                    
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


# # # - - - LINEARIZATION MATRIX - - - # # #
        
# # # - - - WITH ADAPTATION - turing-linearization matrix - - - # # #
"""

!!!!  TO DO  !!!!

def turing_adap_A13(k, be, params):
    return -(1/params.tau_e)*params.b*derivF_e(be)

def turing_adap_A23(k, params):
    return 0

def turing_adap_A31(k, ue, params):
    return (1/params.tau_a)*derivF_a(ue)

def turing_adap_A32(k, params):
    return 0 

def turing_adap_A33(k, params):
    return -(1/params.tau_a)

def turing_adap_A(k, a_ee, a_ei, a_ie, a_ii, be, ue, ui, params):
    #
    Input:
    :k: wavenumber k, array, non-zero
    :a_jl: a_jk Values from function above, determined by fixedpoint (ue, ui), all scalars
    :be: input to F_e for fixed point (ue, ui, derivF_a(ue)), scalar
    :ue: excitatory fixed point value, scalar
    :ui: inhibitory fixed point value, scalar
    :params: dictionary of parameter setting
    
    Output:
    :matrix A: linearization matrix for the coupled IDE-system 3-dimensional (with adaptation) -> 3x3 matrix
    

    #
    
    be = b_e = params.w_ee*ue - params.w_ei*ui - params.b*F_a(ue, params) + params.I_e
    return [[turing_A11(k, a_ee, params), turing_A12(k, a_ei, params), turing_adap_A13(k, be, params)],
            [turing_A21(k, a_ie, params), turing_A22(k, a_ii, params), turing_adap_A23(k, params)],
            [turing_adap_A31(k, ue, params), turing_adap_A32(k, params), turing_adap_A33(k, params)]]
            
"""

# # # - - - TRACE - - - # # #
    
def tr(k, a_ee, a_ii, params):
    return turing_A11(k, a_ee, params) + turing_A22(k, a_ii, params)

def dtr(k, a_ee, a_ii, params):
    return (1/params.tau_e)*a_ee*deriv_f_kernel(params.sigma_e, k, params.kernel)-(1/params.tau_i)*a_ii*deriv_f_kernel(params.sigma_i, k, params.kernel)
    
    
# # # - - - DETERMINANT - - - # # # 
#NOTE: calling the wrong fps (voltage vs activity) can turn the determinante upside done
def det(k, a_ee, a_ei, a_ie, a_ii, params):
    
    return (turing_A11(k, a_ee, params)*turing_A22(k, a_ii, params))-(turing_A12(k, a_ei, params)*turing_A21(k, a_ie, params))

def ddet(k, a_ee, a_ei, a_ie, a_ii, params):
    return (- a_ii * deriv_f_kernel(params.sigma_i, k, params.kernel)
            + a_ee * deriv_f_kernel(params.sigma_e, k, params.kernel)
            - a_ee * a_ii * deriv_f_kernel(params.sigma_e, k, params.kernel) * f_kernel(params.sigma_i, k, params.kernel)
            - a_ee * a_ii * f_kernel(params.sigma_e, k, params.kernel) * deriv_f_kernel(params.sigma_i, k, params.kernel)
            + a_ei * a_ie * deriv_f_kernel(params.sigma_e, k, params.kernel) * f_kernel(params.sigma_i, k, params.kernel)
            + a_ei * a_ie * f_kernel(params.sigma_e, k, params.kernel) * deriv_f_kernel(params.sigma_i, k, params.kernel))
    
    
# # # - - - the functions to test for turing instability - - - # # #

def pos_det(a_ee, a_ei, a_ie, a_ii, params):
    if det(0, a_ee, a_ei, a_ie, a_ii, params)>0:
        return True
    else: 
        return False
    
def neg_tr(k, a_ee, a_ii, params):
    if all(tr(k, a_ee, a_ii, params)< 0): #-0.1**16):
        return True
    else:
        return False
    
def det_traj(k, a_ee, a_ei, a_ie, a_ii, params):
    if any(det(k, a_ee, a_ei, a_ie, a_ii, params)< 0): #-0.1**16):
        return True
    else:
        return False
    
def lmbd(k_real, a_ee, a_ei, a_ie, a_ii, params):
    k = k_real.astype(complex)
    lmbd_plus = (1/2)*(tr(k, a_ee, a_ii, params) + np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    lmbd_minus = (1/2)*(tr(k, a_ee, a_ii, params) - np.sqrt(tr(k, a_ee, a_ii, params)**2 - 4*det(k, a_ee, a_ei, a_ie, a_ii, params)))
    return [lmbd_plus, lmbd_minus]



    
# # # - - - the functions to check Turing instability and check possibility of spatiotemporal patterns - - - # # #

def violationType(k, a_ee, a_ei, a_ie, a_ii, params):
    
    """
        This function checks with what condition of the linear stability analysis on a continuum is violated.
        (i.e. run this check only for fixed points that are linearly stable in the local [i.e. one-node] system)
        Options are: det(A(k)) > 0 is violated for a k0!=0 (i.e. det(A(k0))=0). Then we have a static Turing bifurcation point (i.e. spatial pattern). This is equivalent to im(\lambda)=0.
                     tr(A(k))  < 0 is violated for a k0!=0 (i.e. tr(A(k0))=0). Then we speak of a dynamic Turing bifurcation point (i.e. spatiotemporal pattern)
                     
        For the output-wavenumber k0, we have that "The Turing bifurcation point is defined by the smalles non-zero wave number k0" 
            - by Meijer & Commbes: Travelling waves in a neural field model with refractoriness (2013)
                     
        Input:
        :k: wavenumber (array)
        :a_kj: Values necessary to determine det & tr
        :params: parameter setting of model
        
        Output: 
        :k0: wavenumber k for which the violation appears
        :violation_type: type of violation. Options are: 0 (no violation), 1 (static), 2 (dynamic), 3 (both).
    """
    
    violation_type = 0
    k0 = None
    
    if not pos_det(a_ee, a_ei, a_ie, a_ii, params):
        return 0
    if det_traj(k, a_ee, a_ei, a_ie, a_ii, params):
      #  determ = det(k, a_ee, a_ei, a_ie, a_ii, params)
      #  k00 = determ[np.abs(determ - 0).argmin()]
      #  kindex = np.linspace(0, len(k)-1, len(k))
        k0 = determinantTuringBifurcation(a_ee, a_ei, a_ie, a_ii, params)[0]
        violation_type = 1
    elif not neg_tr(k, a_ee, a_ii, params):
        #return smallest non-zero k0!
       # tracy = tr(k, a_ee, a_ei, a_ie, a_ii, params)
       # k00 = tracy[np.abs(tracy - 0).argmin()]
       # kindex = np.linspace(0, len(k)-1, len(k))
        k0 = traceTuringBifurcation(a_ee, a_ii, params)[0] #k[int(kindex[list(tracy).index(k00)])]
        violation_type = 2
    elif det_traj(k, a_ee, a_ei, a_ie, a_ii, params) and not neg_tr(k, a_ee, a_ii, params):
        violation_type = 3
    
    return violation_type, k0


def determinantTuringBifurcation(a_ee, a_ei, a_ie, a_ii, params):
    """ Determine the smallest non-zero wave number k0 that defines the Turing bifurcation point for which we have the condition 
        det(A(k))>0 violated."""
    
    k00=[]
    

    for k in np.linspace(0, 2, 11):
        sol = root(det, k, args=(a_ee, a_ei, a_ie, a_ii, params,), method='lm')
       # fix_point = [round(sol.x[0], 8), round(sol.x[1], 8)]
        if sol.success:
            closeness = all(np.isclose(det(sol.x, a_ee, a_ei, a_ie, a_ii, params), 0.0))
            if closeness:
                if len(k00)==0: #always append the firstly derived fixed point
                    k00.append(sol.x)
                else:
                    already_derived = False
                    for i in range(len(k00)):
                        if all(np.isclose(sol.x, k00[i], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                    if already_derived:
                        pass #skip the already derived fixed points
                    else:
                        k00.append(sol.x)
    
    return min(abs(np.array(k00)))

def traceTuringBifurcation(a_ee, a_ii, params):
    """ Determine the smallest non-zero wave number k0 that defines the Turing bifurcation point for which we have the condition 
        det(A(k))>0 violated."""
    
    k00=[]
    

    for k in np.linspace(0, 2, 11):
        sol = root(tr, k, args=(a_ee, a_ii, params,), method='lm')
       # fix_point = [round(sol.x[0], 8), round(sol.x[1], 8)]
        if sol.success:
            closeness = all(np.isclose(tr(sol.x, a_ee, a_ii, params), 0.0))
            if closeness:
                if len(k00)==0: #always append the firstly derived fixed point
                    k00.append(sol.x)
                else:
                    already_derived = False
                    for i in range(len(k00)):
                        if all(np.isclose(sol.x, k00[i], atol=1e-9)):
                            already_derived = True
                        else: 
                            pass
                    if already_derived:
                        pass #skip the already derived fixed points
                    else:
                        k00.append(sol.x)
    
    return min(abs(np.array(k00)))

