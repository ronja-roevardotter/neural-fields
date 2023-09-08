import numpy as np
import py.twoD.kernels2d as ks

"""
Note: any function here can be called speratly, but one has to use a dotdict-dictionary to call the functions and know, which parameters are used for computations.
"""
    
    

class dotdict(dict):
    """dot.notation access to dictionary attributes. dotdict = GEKLAUT von neurolib.utils.collections.py 
    (https://github.com/neurolib-dev/neurolib/blob/master/neurolib/utils/collections.py)
    Example:
    ```
    model.params['duration'] = 10 * 1000 # classic key-value dictionary
    model.params.duration = 10 * 10000 # easy access via dotdict
    ```
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    # now pickleable!!!
    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)
  
    
def defaultParams():
    
    params = dotdict({})
    
    #default model-type is activity-based i.e. Wilson-Cowan
    params.mtype = 'activity'
    
    #membrane time constants [no units yet]
    params.tau_e = 1.0 #2.5 
    params.tau_i = 1.5 #3.75
    
    #coupling weights (determining the dominant type of cells, e.g. locally excitatory...)
    params.w_ee = 3.2 #excitatory to excitatory
    params.w_ei = 2.6 #inhibitory to excitatory
    params.w_ie = 3.3 #excitatory to inhibitory
    params.w_ii = 0.9 #inhibitory to inhibitory
    
    #threshold and gain factors for the sigmoidal activation functions 
    params.beta_e = 5 #excitatory gain
    params.beta_i = 5 #inhibitory gain
    params.mu_e = 0 #excitatory threshold
    params.mu_i = 0 #inhibitory threshold
    
    # # - - adaptation parameters - - # #
    
    #transfer function
    params.beta_a = 5
    params.mu_a = 0
    
    #strength and time constant - to turn adaptation off: set b=0
    params.b = 0 #0.5 - set it 0 until further notice (mostly to not accidentally run analysis with adaptation)
    params.tau_a = 600
    
    # # - - - - # #
    
    
    #Write seperate function for setting the parameters of the coupling function w(x), but set the function type:
    params.kernel = 'gaussian' #else choosable: exponential,...

    #for simplification:
    params.sigma_e = 1 #characterizes the spatial extent of the excitatory coupling
    params.sigma_i = 3 #characterizes the spatial extent of the inhibitory coupling
    
    
    #external input currents: oscillatory state for default params
    params.I_e = 0.0 
    params.I_i = 0.0

    #temporal 
    #choose number of integration time step in seconds
    params.dt = 0.1 #[ms] : we assume ms per one unit-time => 0.1 is every 10th millisecond
    
    #choose time interval [start,end]=[0,T] [in ms]
    params.start_t = 0
    params.end_t = 2000
    
    #spatial
    #choose number of pixels per axis: n=#nodes on x-axis (hosizontal), m=#nodes on y-axis (vertical)
    params.n = 128
    params.m = 128
    
    #choose spatial boundaries (intervals of spatial spread)
    params.xlength = 50 #length of spatial component [for delay computations assumed to be in mm]
    params.ylength = 50 #length of spatial component [for delay computations assumed to be in mm]
    
    params.c = 10 #m/s velocity of activity in m/s
    
    #set the amount of images you want to save during integration
    params.pic_nmb = 10
    
    return params

    
def setTime(params):
    """
    Necessary params:
    :start_t: beginning of time
    :end_t: end of time
    :dt: integration time step
    
    returns: array time of time intervall
    """

    return np.arange(params.start_t, params.end_t + params.dt, params.dt)

def setTimeStamps(params):
    """
    Necessary params:
    :start_t: beginning of time
    :time: time array
    :dt: integration time step
    
    returns: array time of time intervall
    """
    time_stamps = np.linspace(params.start_t, int(params.time[-1]*(1/params.dt)), int(params.pic_nmb+1)).astype(int)
    return time_stamps

def setSpace(params, shift=False):
    """
    Necessary params: 
    :length: length of ring circumference
    :n: number of pixels
    
    returns: x, dx
    :x: distance array from one pixel to all other pixels
    :dx: float of integration space step
    """
    
    x_inf, x_sup, dx= -(params.xlength/2), +(params.xlength/2), (params.xlength/float(params.n))
    y_inf, y_sup, dy= -(params.ylength/2), +(params.ylength/2), (params.ylength/float(params.m))
        
    x = np.arange(x_inf, x_sup, dx)
    y = np.arange(y_inf, y_sup, dy)
        
        
    if shift:
        x = np.fft.fftshift(x)
        y = np.fft.fftshift(y)
        return x, y, dx, dy
    else:
        return x, y, dx, dy
    
    
    
def ringValues(params):
    """
    Necessary parameters:
    :kernel: string, which spatial kernel shall be used (gaussian vs exponential)
    :sigma_e: float, excitatory spread
    :sigma_i: float, inhibitory spread
    :x: distance array
    :dx: float, integration space step
    
    returns: ke, ki
    :ke: array of by spatial kernel function weighted excitatory connectivity to others
    :ki: array of by spatial kernel function weighted inhibitory connectivity to others
    """
    
    kernel_func = getattr(ks, params.kernel)
    
    ke = kernel_func(params.sigma_e, params.distx)*params.dx*params.dy
    ki = kernel_func(params.sigma_i, params.distx)*params.dx*params.dy
    
    #normalize kernel w.r.t. integration space step dx
 #   alpha_e = np.sum(ke)*params.dx*params.dy #normalisation factor for exc
 #   alpha_i = np.sum(ki)*params.dx*params.dy #normalisation factor for inh
    
    #normalisation & consideration of integration step s.t. we don't have to consider that anymore later.
 #   ke = ke*params.dx*params.dy #(params.dx/alpha_e) 
 #   ki = ki*params.dx*params.dy #(params.dx/alpha_i)
    
 #   fourier_func = getattr(ks, 'f_' + params.kernel)
    
 #   ke_fft = (1/(2*np.pi)) * fourier_func(params.sigma_e, params.xcoords, params.ycoords)
 #   ki_fft = (1/(2*np.pi)) * fourier_func(params.sigma_i, params.xcoords, params.ycoords)
    
    return ke, ki #, ke_fft, ki_fft

def setDelay(params):
    """ 
    Necessary params:
    :x: distance array
    :c: velocity of activity in [m/s]
    
    returns:
    :delay: array of integers that indicate the delay from pixel to pixel based on velocity c
    """
    
    delay_x = abs(params.x) #assume distance in mm and velocity in m/s -> transform c into mm/s by multiplying with 1000
    params.c *= 1000
    delay = delay_x*(1/params.c) # in seconds
    
    delay *= 1000 #in milliseconds (1sec = 1000ms)
    delay = delay*(1/params.dt)# in time steps
    
    delay += 1
    delay = abs(delay.astype(int))
    
    return delay
    
    
def setParams(pDict):
    
    params = defaultParams()
    
    if pDict is not None:
        for k, val in zip(pDict.keys(), pDict.values()):
            params[k] = val
                
    params.time = setTime(params)
    
    params.time_stamps = setTimeStamps(params)
    
    params.x, params.y, params.dx, params.dy = setSpace(params, shift=False)
    
    params.xcoords, params.ycoords = np.meshgrid(params.x, params.y)
    
    #the distance matrix w.r.t. the zero-point 
    #we also define the activity on this grid. We observe the grid w.r.t. distance always.
    params.distx = params.xcoords**2 + params.ycoords**2
    
    params.ke, params.ki = ringValues(params) #, params.ke_fft, params.ki_fft = ringValues(params)
    
    params.ke_fft = np.fft.fft2(params.ke) # np.fft.fft2(np.fft.fftshift(params.ke)) #
    params.ki_fft = np.fft.fft2(params.ki) # np.fft.fft2(np.fft.fftshift(params.ki)) #
    
    params.delay = setDelay(params)
                
    return params

    