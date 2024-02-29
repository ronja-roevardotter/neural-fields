import numpy as np

def alternatingSeed(array, seed_amp):
    array[1::2, :] -= seed_amp
    array[::2, :] += seed_amp
    return array

def kernSeed(array, kernel, seed_amp):
    array += kernel*seed_amp
    return array


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
    
    # # - - adaptation parameters - - # #
    
    #transfer function
    beta_a = params.beta_a
    mu_a = params.mu_a
    
    #strength and time constant
    b = params.b
    tau_a = params.tau_a
    
    # # - - - - # #
    
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
    dy = params.dy
    time = params.time
    time_stamps = params.time_stamps
    
    ke = params.ke
    ki = params.ki
    
    ke_fft = params.ke_fft
    ki_fft = params.ki_fft
    
    delay = params.delay
    
    comparison = fp==[0.0,0.01]
    
    seed =  params.seed
    seed_amp =  params.seed_amp
    seed_func = params.seed_func 
        
    
    if all(comparison):
        init_exc = fp
        init_inh = fp
        init_adaps = fp
    else:
        a_fp = 1/(1+np.exp(-beta_a*(fp[0]-mu_a)))
        init_exc = [fp[0]-0.1*(10**(-10)), fp[0]+0.1*(10**(-10))]
        init_inh = [fp[1]-0.1*(10**(-10)), fp[1]+0.1*(10**(-10))]
        init_adaps = [a_fp-0.1*(10**(-10)), a_fp+0.1*(10**(-10))]
    
    if seed and not all(comparison): 
        ue_init=np.ones(ke.shape)*fp[0]
        ui_init=np.ones(ki.shape)*fp[1]
        adaps_init=np.ones(ke.shape)*(1/(1+np.exp(-beta_a*(fp[0]-mu_a))))

        if seed_func == 'sinus':
            ue_init = alternatingSeed(ue_init, seed_amp)
            ui_init = alternatingSeed(ui_init, seed_amp)
            adaps_init = alternatingSeed(adaps_init, seed_amp)
        
            ue_init = alternatingSeed(ue_init.T, seed_amp)
            ui_init = alternatingSeed(ui_init.T, seed_amp)
            adaps_init = alternatingSeed(adaps_init.T, seed_amp)
        elif seed_func == 'kern':
            ue_init = kernSeed(ue_init, ke, seed_amp)
            ui_init = kernSeed(ui_init, ki, seed_amp)
            adaps_init = kernSeed(adaps_init, ke, seed_amp)
            
    else:
        #the initialisation I have to make to start the integration
        ue_init=np.random.uniform(init_exc[0], init_exc[1], ke.shape) #-> shape = (m,n)
        ui_init=np.random.uniform(init_inh[0], init_inh[1], ki.shape)
        adaps_init=np.random.uniform(init_adaps[0], init_adaps[1], ke.shape) #-> since exc is shape giver for adaps
    
 #   print('ue_init=%s' %str(ue_init))
    
    integrate = globals()[itype]
    
    ue, ui =  integrate(mtype,
                        tau_e, tau_i,
                        w_ee, w_ei, w_ie, w_ii,
                        beta_e, beta_i, mu_e, mu_i,
                        I_e, I_i,
                        beta_a, mu_a, b, tau_a, adaps_init, 
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
             beta_a, mu_a, b, tau_a, adaps, 
             dt, time, time_stamps, delay, 
             n, length, c, x, dx, 
             ke, ki, ke_fft, ki_fft,
             ue, ui):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    ue_old = np.copy(ue)
    ui_old = np.copy(ui)
    
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




# # # - - - WITH ADAPTATIOn: Integration by Fourier Transform, Product, Inverse Fourier Transform of convolution - - - # # #

def inte_adaptation(mtype,
             tau_e, tau_i,
             w_ee, w_ei, w_ie, w_ii,
             beta_e, beta_i, mu_e, mu_i,
             I_e, I_i,
             beta_a, mu_a, b, tau_a, adaps, 
             dt, time, time_stamps, delay, 
             n, length, c, x, dx, 
             ke, ki, ke_fft, ki_fft,
             ue, ui):
    
    def Fe(x):
        return 1/(1+np.exp(-beta_e*(x-mu_e)))
    
    def Fi(x):
        return 1/(1+np.exp(-beta_i*(x-mu_i)))
    
    def Fa(x):
        return 1/(1+np.exp(-beta_a*(x-mu_a)))
    
    ue_old = np.copy(ue)
    ui_old = np.copy(ui)
    adaps_old = np.copy(adaps)
    
    ue_out = []
    ui_out = []
    adaps_out = []
    
    ue_out.append(ue_old.copy())
    ui_out.append(ui_old.copy())
    adaps_out.append(adaps_old.copy())
    
    print('adaps shape=%s, e-shae=%s, inh-shape=%s' %(str(adaps.shape), str(ue.shape), str(ui.shape)))
    
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
            rhs_adaps = ((1/tau_a)*(-adaps_old + Fa(ue_old)))
            rhs_e = ((1/tau_e)*(-ue_old + Fe(w_ee*conv_e - w_ei*conv_i - b*adaps_old + I_e)))
            rhs_i = ((1/tau_i)*(-ui_old + Fi(w_ie*conv_e - w_ii*conv_i + I_i)))
        else:
            rhs_e = ((1/tau_e)*(-ue_old + w_ee*conv_e - w_ei*conv_i + I_e))
            rhs_i = ((1/tau_i)*(-ui_old + w_ie*conv_e - w_ii*conv_i + I_i))
        
        
        #integrate with euler integration
        ue_new = ue_old + (dt * rhs_e)
        ui_new = ui_old + (dt * rhs_i)
        adaps_new = adaps_old + (dt * rhs_adaps)
        
        if t in time_stamps:
            ue_out.append(ue_new.copy())
            ui_out.append(ui_new.copy())
            adaps_out.append(adaps_new.copy())
            print('Round t=%i with ue-shape=%s' %(int(t),str(ue_new.shape)))
            
        ue_old = ue_new
        ui_old = ui_new
        adaps_old = adaps_new
        
    
    return ue_out, ui_out


