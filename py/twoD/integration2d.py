import numpy as np


def runIntegration(params, fp=np.array([0.0, 0.01]), itype='inte_fft',):
    
    """
    Before we can run the integration-loop, we have to set the parameters and call the integration by them 
    s.t. nothing's accidentally overwritten.
    """
    
    mtype = params.mtype
    
    #membrane time constants [no units yet]
    tau_e = params.tau_e
    tau_i = params.tau_i
    
    #coupling weights (determining the dominant type of cells, e.g. locally excitatory...)
    w_ee = params.w_ee
    w_ei = params.w_ei
    w_ie = params.w_ie
    w_ii = params.w_ii
    
    #threshold and gain factors for the sigmoidal activation functions 
    beta_e = params.beta_e
    beta_i = params.beta_i
    mu_e = params.mu_e
    mu_i = params.mu_i
    
    #external input currents: oscillatory state for default params
    I_e = params.I_e
    I_i = params.I_i

    #temporal 
    dt = params.dt
    
    #spatial
    n = params.n
    length = params.length
    c = params.c
    
    x = params.x
    dx = params.dx
    time = params.time
    time_stamps = params.time_stamps
    
    ke = params.ke
    ki = params.ki
    
    ke_fft = params.ke_fft
    ki_fft = params.ki_fft
    
    delay = params.delay
    
    
    comparison = fp==[0.0,0.01]
    
    if all(comparison):
        init_exc = fp
        init_inh = fp
    else:
        init_exc = [fp[0]-0.1*(10**(-8)), fp[0]+0.1*(10**(-8))]
        init_inh = [fp[1]-0.1*(10**(-8)), fp[1]+0.1*(10**(-8))]
    
    #the initialisation I have to make to start the integration
    ue_init=np.random.uniform(init_exc[0], init_exc[1], ke.shape) #-> shape = (m,n)
    ui_init=np.random.uniform(init_inh[0], init_inh[1], ki.shape)
    
 #   print('ue_init=%s' %str(ue_init))
    
    integrate = globals()[itype]
    
    ue, ui =  integrate(mtype,
                        tau_e, tau_i,
                        w_ee, w_ei, w_ie, w_ii,
                        beta_e, beta_i, mu_e, mu_i,
                        I_e, I_i,
                        dt, time, time_stamps, delay, 
                        n, length, c, x, dx, 
                        ke, ki, ke_fft, ki_fft,
                        ue_init, ui_init)
    
    
    return ue, ui

# # # - - - Integration by Fourier Transform, Product, Inverse Fourier Transform of convolution - - - # # #

def inte_fft(mtype,
             tau_e, tau_i,
             w_ee, w_ei, w_ie, w_ii,
             beta_e, beta_i, mu_e, mu_i,
             I_e, I_i,
             dt, time, time_stamps, delay, 
             n, length, c, x, dx, 
             ke, ki, ke_fft, ki_fft,
             ue_init, ui_init):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    ue_old = np.copy(ue_init)
    ui_old = np.copy(ui_init)
    
    ue_out = []
    ui_out = []
    
    ue_out.append(ue_old.copy())
    ui_out.append(ui_old.copy())
    
    for t in range(1,int(len(time))): 
        
        #indeces for delays - makes the delayed time steps easier to call
        #indeces = np.array([t*np.ones(n)-delay]).astype(int)
        
        if mtype=='activity':
            ve = np.fft.fft2(ue_old)
            vi = np.fft.fft2(ui_old)
        else:
            ve = np.fft.fft2(Fe(ue_old))
            vi = np.fft.fft2(Fi(ui_old))
        
        Le = ke_fft * ve
        Li = ki_fft * vi
        
        conv_e = np.fft.ifft2(Le).real
        conv_i = np.fft.ifft2(Li).real
        
        
        #determine the RHS before integrating over it w.r.t. time t
        if mtype=='activity':
            rhs_e = ((1/tau_e)*(-ue_old + Fe(w_ee*conv_e - w_ei*conv_i + I_e)))
            rhs_i = ((1/tau_i)*(-ui_old + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        else:
            rhs_e = ((1/tau_e)*(-ue_old + w_ee*conv_e - w_ei*conv_i + I_e))
            rhs_i = ((1/tau_i)*(-ui_old + w_ie*conv_e - w_ii*conv_i + I_i))
        
        
        #integrate with euler integration
        ue_new = ue_old + (dt * rhs_e)
        ui_new = ui_old + (dt * rhs_i)
        
        if t in time_stamps:
            ue_out.append(ue_new.copy())
            ui_out.append(ui_new.copy())
        #    print('Round t=%i' %int(t))
            
        ue_old = ue_new
        ui_old = ui_new

        
    
    return ue_out, ui_out

