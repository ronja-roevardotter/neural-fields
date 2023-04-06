import numpy as np
import py.kernels as ks

from py.analysis import a_jkValues


def runIntegration(params, fp=np.array([0.0, 0.01])):
    
    """
    Before we can run the integration-loop, we have to set the parameters and call the integration by them 
    s.t. nothing's accidentally overwritten.
    """
    
    mtype = params.mtype
    
    tau_e = params.tau_e
    tau_i = params.tau_i
    
    #values of homogeneous steady state that forms the new ODE:
    exc = fp[0]
    inh = fp[1]
    
    a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
    
    #spatial
    n = params.n
    length = params.length
    c = params.c
    
    omega = np.linspace(-length/2, length/2, n)
    time = params.time
    dt = params.dt
    
    kernel_func = getattr(ks, 'f_'+params.kernel)
    
    ke_hat = kernel_func(params.sigma_e, omega)
    ki_hat = kernel_func(params.sigma_i, omega)
    
    #Alternative:
   # ke_hat = params.ke_fft
   # ki_hat = params.ki_fft
    
    delay = params.delay
    
   # comparison = fp==[0.0,0.01]
    
   # if all(comparison):
    init_exc = fp
    init_inh = fp
    #else:
    #    init_exc = [fp[0]-0.1*(10**(-6)), fp[0]+0.1*(10**(-6))]
    #    init_inh = [fp[1]-0.1*(10**(-6)), fp[1]+0.1*(10**(-6))]
    
    #the initialisation I have to make to start the integration
    phie_init = np.zeros((len(time),n)) #leads to [rows, columns] = [time, pixels (space)]
    PHII_init = np.zeros((len(time),n))
    phie_init[0]=np.random.uniform(init_exc[0], init_exc[1], n)
    PHII_init[0]=np.random.uniform(init_inh[0], init_inh[1], n)
    
    ue, ui =  integration(mtype, tau_e, tau_i,
                          exc, inh,
                          a_ee, a_ei, a_ie, a_ii,
                          dt, time, delay, 
                          n, length, c,
                          ke_hat, ki_hat,
                          phie_init, PHII_init)
    
    
    return ue, ui

# # # - - - Integration by Fourier Transform, Product, Inverse Fourier Transform of convolution - - - # # #

def integration(mtype, tau_e, tau_i,
                exc, inh,
                a_ee, a_ei, a_ie, a_ii,
                dt, time, delay, 
                n, length, c,
                ke_hat, ki_hat,
                phie_init, PHII_init):
    
    ue = np.copy(phie_init)
    ui = np.copy(PHII_init)
    
    for t in range(1,int(len(time))): 
        
        #determine the RHS before integrating over it w.r.t. time t
        rhs_e = ((1/tau_e)*(-ue[t-1] + a_ee*ke_hat*ue[t-1] - a_ei*ki_hat*ui[t-1]))
        rhs_i = ((1/tau_i)*(-ui[t-1] + a_ie*ke_hat*ue[t-1] - a_ii*ki_hat*ui[t-1]))
        
        
        #integrate with euler integration
        ue[t] = ue[t-1] + (dt * rhs_e)
        ui[t] = ui[t-1] + (dt * rhs_i)
        
    
    return ue, ui