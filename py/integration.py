import numpy as np


def runIntegration(params, fp=np.array([0.0, 0.01]), itype='int_fft',):
    
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
        init_exc = [fp[0]-0.1*(10**(-6)), fp[0]+0.1*(10**(-6))]
        init_inh = [fp[1]-0.1*(10**(-6)), fp[1]+0.1*(10**(-6))]
    
    #the initialisation I have to make to start the integration
    ue_init = np.zeros((len(time),n)) #leads to [rows, columns] = [time, pixels (space)]
    ui_init = np.zeros((len(time),n))
    ue_init[0]=np.random.uniform(init_exc[0], init_exc[1], n)
    ui_init[0]=np.random.uniform(init_inh[0], init_inh[1], n)
    
    
    
    integrate = globals()[itype]
    
    ue, ui =  integrate(mtype,
                        tau_e, tau_i,
                        w_ee, w_ei, w_ie, w_ii,
                        beta_e, beta_i, mu_e, mu_i,
                        I_e, I_i,
                        dt, time, delay, 
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
             dt, time, delay, 
             n, length, c, x, dx, 
             ke, ki, ke_fft, ki_fft,
             ue, ui):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    ve = np.fft.fft(np.copy(ue[0]))
    vi = np.fft.fft(np.copy(ui[0]))
    
    for t in range(1,int(len(time))): #kann das -1 hier und oben nicht weg? 
        
        if mtype == 'activity':
            Le = ke_fft * ve
            Li = ki_fft * vi
        else:
            Le = ke_fft * Fe(ve)
            Li = ki_fft * Fi(vi)
        
        Le = np.fft.ifft(Le)
        Li = np.fft.ifft(Li)
        
        conv_e = Le.real
        conv_i = Li.real
        
        
        #determine the RHS before integrating over it w.r.t. time t
        if mtype=='activity':
            rhs_e = ((1/tau_e)*(-ue[t-1] + Fe(w_ee*conv_e - w_ei*conv_i + I_e)))
        
            rhs_i = ((1/tau_i)*(-ui[t-1] + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        else:
            rhs_e = ((1/tau_e)*(-ue[t-1] + w_ee*conv_e - w_ei*conv_i + I_e))
        
            rhs_i = ((1/tau_i)*(-ui[t-1] + w_ie*conv_e - w_ii*conv_i + I_i))
        
        
        #integrate with euler integration
        ue[t] = ue[t-1] + (dt * rhs_e) #u_e[t-1] + (self.md.variables.dt * rhs_e)
        ui[t] = ui[t-1] + (dt * rhs_i) #u_i[t-1] + (self.md.variables.dt * rhs_i)
        
        v_e = np.fft.fft(np.copy(ue[t]))
        v_i = np.fft.fft(np.copy(ui[t]))
    
    return ue, ui

# # # - - - Integration with integral approximation of convolution - - - # # #

def inte_approxi(mtype,
                 tau_e, tau_i,
                 w_ee, w_ei, w_ie, w_ii,
                 beta_e, beta_i, mu_e, mu_i,
                 I_e, I_i,
                 dt, time, delay, 
                 n, length, c, x, dx, 
                 ke, ki, ke_fft, ki_fft,
                 ue, ui):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    d_max = max(delay)
    indices = np.linspace(0,n-1, n).astype(int)
    
    ke_mtx = np.zeros((n,n))
    ki_mtx = np.zeros((n,n))
    delay_mtx = np.zeros((n,n)).astype(int)
    
    for j in range(n):
        ke_mtx[j] = np.roll(ke, j)
        ki_mtx[j] = np.roll(ki, j)
        delay_mtx[j] = np.roll(delay, j).astype(int)
    
    for t in range(1,int(len(time))): 
        
        L_e=np.zeros(n)
        L_i=np.zeros(n)
        
        if mtype=='activity':
            if t<=d_max+1:
                
                for j in range(n):
                   # print('kernel=%s and shape=%s: ' %((str(ke[j]),str(ke[j].shape))))
                   # print('ue=%s and shape=%s: ' %((str(ue),str(ue.shape))))
                    L_e[j] = (ke_mtx[j] @ ue[t-1])
                    L_i[j] = (ki_mtx[j] @ ui[t-1])
            else:
                for j in range(n):
                    temp_e = ue[t-delay_mtx[j], indices]
                    temp_i = ui[t-delay_mtx[j], indices]
                    L_e[j] = (ke_mtx[j] @ temp_e)
                    L_i[j] = (ki_mtx[j] @ temp_i)
        else:
            if t<=d_max+1:
                for j in range(n):
                    L_e[j] = ke_mtx[j] @ Fe(ue[t-1])
                    L_i[j] = ki_mtx[j] @ Fi(ui[t-1])
            else:
                for j in range(n):
                    temp_e = ue[t-delay_mtx[j], indices]
                    temp_i = ui[t-delay_mtx[j], indices]
                    L_e[j] = ke_mtx[j] @ Fe(temp_e)
                    L_i[j] = ki_mtx[j] @ Fi(temp_i)
                    
            
        conv_e = L_e
        conv_i = L_i
        
        
        #determine the RHS before integrating over it w.r.t. time t
        if mtype=='activity':
            rhs_e = ((1/tau_e)*(-ue[t-1] + Fe(w_ee*conv_e - w_ei*conv_i + I_e)))
            rhs_i = ((1/tau_i)*(-ui[t-1] + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        else:
            rhs_e = ((1/tau_e)*(-ue[t-1] + w_ee*conv_e - w_ei*conv_i + I_e))
            rhs_i = ((1/tau_i)*(-ui[t-1] + w_ie*conv_e - w_ii*conv_i + I_i))
        
        
        #integrate with euler integration
        ue[t] = ue[t-1] + (dt * rhs_e)
        ui[t] = ui[t-1] + (dt * rhs_i)
        
    return ue, ui