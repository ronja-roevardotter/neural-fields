import numpy as np

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
    inside = (params.w_ee * ue + params.I_e- inverseF_e(ue, params))
    return (1/params.w_ei) * inside

def activity_u_e(ui, params):
    """Returns the inhibitory nullcline w.r.t. ui (\frac{ui}{dt}=0)
       for the activity-based model"""
    inside = (inverseF_i(ui, params) + params.w_ii * ui - params.I_i)
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
            sol = root(activity, [i, i], args=(params,), jac=activity_A, method='lm')
            print('solution to root: ', sol.x)
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
    return fixed_points

#function to determine the stability of fixed points, depending on the fixed points AND model-type

                    
def checkFixPtsStability(fixed_points, params):
    
    fixed_points = np.sort(fixed_points, axis=0)
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
                    


def runAll():
    #generates the fixed_points-list of all fixed points
    computeFPs() #has no input, since all necessary parameters are set in __init__()

    checkFixPtsStability(fixed_points) #generates the stability list with 1 for stable 0 otherwise
    
    if sum(stability)==2:
        fixed_points = np.sort(fixed_points, axis=0)
        bistable = True

