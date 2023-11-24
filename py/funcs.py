import numpy as np
import scipy.signal as signal 


def getSwitchArray(array):
    """ returns the array at which positions the x-axis (have a zero) is crossed
    help for implementation found on 'stackoverflow.com'
    INPUT:
    :array: 1d array of numerical values
    
    OUPUT:
    :sign_switch_array: array with 1 at position of switch
    e.g. array=[2,3,1,-1,-2,-1,1,2] -> sign_switch_array = [0, 0, 0, 1, 0, 0, 1, 0]
    """
    
    
    signs = np.sign(array)
    sign_switch_array = ((np.roll(signs, 1) - signs) != 0).astype(int)
    
 #   print('getSwitchArray returns this list: %s' %str(sign_switch_array))
    
    return sign_switch_array

def getSwitchIndex(array):
    """ returns the indeces at which the positions the x-axis (have a zero) is crossed
    help for implementation found on 'geeksforgeeks.org'
    INPUT:
    :array: 1d array of numerical values
    
    OUPUT:
    :idx: array with indeces of switch
    e.g. array=[2,3,1,-1,-2,-1,1,2] -> idx = [3,6]
    """
    
    sign_switch_array = getSwitchArray(array)
    idx = np.where(sign_switch_array == 1)[0]
    
    if not idx.any():
        idx = [0]
    elif len(idx)>1 and idx[0]==0:
        idx = idx[1:]
    
    return idx

def getCommonElement(array1, array2):
    array1_set = set(array1)
    array2_set = set(array2)
    if (array1_set & array2_set):
        return True
    else:
        return False
    
    
    

def getPSD(array, fs, maxfreq=300, nperseg = 1):
    """returns the Powerspectrum (density: [V**2/Hz]) with the possibility to cut off the PSD at a maximum frequency
    
    INPUT:
    :array: time series (type: numpy-array)
    :fs: sampling frequency of array (e.g. variables['delta_t'] for array over time, variables['n'] for array over space)
    :maxfreq: maximum frequency to observe
    
    OUTPUT:
    :freqs: array of all frequencies that are looked at for the PDS
    :PSD_den: Power Spectrum Density (i.e. returns the power per frequency over the freqs array)
    """
    
    freqs, Pxx_den = signal.welch(array, fs, window='hann', nperseg=int(nperseg*fs))
    
    if maxfreq==None:
        maxfreq = max(freqs)
    
    freqs = freqs[freqs < maxfreq]
    Pxx_den = Pxx_den[0 : len(freqs)]
    
    return freqs, Pxx_den



def getAvgPSD(arrays, fs, maxfreq=None):
    """Returns the average Power Spectrum Density for a mxn-dimensional array.
    
    INPUT:
    :arrays: mxn-dimensional array (m rows, n columns)
    :fs: sampling frequency
    :maxfreq: maximum frequency
    
    OUTPUT:
    :freqs: array of all frequencies that are looked at for the avg-PSD
    :avg_Pxx_den: average PSD (averages over rows)
    
    e.g. I give the array exc with dimension 37x4000
    then the PSD will be computed for frequencies over time and averaged over rows, which are the nodes (space)
    => returned avg_Pxx_den indicates the power per frequencies over time averaged over each node
    i.e. in that case, if the avg_Pxx_den is close to zero everywhere, I do not have a change in activity over time => temporally homogeneous
    
    """
    
    freqs, temp_Pxx_den = getPSD(arrays[0], fs, maxfreq)
    
    all_Pxx_den = np.zeros((int(arrays.shape[0]),(len(temp_Pxx_den))))
    
    for idx, array in enumerate(arrays[1:]):
        f, Pxx_den = getPSD(array, fs, maxfreq)
        all_Pxx_den[idx+1] = Pxx_den
        
    avg_Pxx_den = np.mean(all_Pxx_den, axis=0)
        
    
    return freqs, avg_Pxx_den


def getPosition(pixel_nmb, params):
    """
    This function returns the position on the ONE-dimensional line of a certain pixel.
    Input:
    :pixel_nmb: number of the pixel, integer
    :params: only necessary - params.x
    
    Output:
    :position: position x on line.
    """
    
    position = params.x[pixel_nmb]
    
    return position


            # # # - - -                                                                - - - # # # 
# # # - - - The following functions were originally added to determine the phase & phase latencies - - - # # #
            # # # - - -                                                                - - - # # # 

def rotation_in_latency(array):
    """This function determines, based on the phase latency, into which direction the traveling waves rotate.
    INPUT:
    :array: numpy array, 1-dimensional, consists of the phase latency 
            ('how many time steps it takes node to cross the threshold 2\pi') per node (i.e. dim(array)=params.n)
            
    OUTPUT:
    :rotation: -1 (clockwise), or +1 (counterclockwise)
    """
    
    max_arg = np.argmax(array)
    argmax_before = max_arg-1
    argmax_after = max_arg+1
    if argmax_after == len(array):
        argmax_after=0
    diff_to_left = np.abs(array[max_arg]-array[int(argmax_before)])
    diff_to_right = np.abs(array[max_arg]-array[int(argmax_after)])
    
    if diff_to_left>diff_to_right:
        rotation = +1 #"counterclockwise"
    else:
        rotation = -1 #"clockwise"

    return rotation

def count_nodes_for_descent(array, rotation):
    """This function determines the amount of nodes that are necessary for one full phase transition from max to min
    i.e. how many nodes are necessary for one full phase
    INPUT: 
    :array: numpy array, 1-dimensional, consists of the phase latency 
            ('how many time steps it takes node to cross the threshold 2\pi') per node (i.e. dim(array)=params.n)
    :rotation: identifies in which direction we have to descent to count until next max
    
    OUTPUT:
    :count: amount of nodes that are necessary until the next full phase transition starts
    """
    
    max_arg = np.argmax(array)
    
    count = 0
    node = max_arg
    if rotation<0:
        while array[node-1] < array[node]:
            count += 1
            node -= 1
            node = int(node)
            if node >= len(array)-1:
                node = 0
    else:
        while array[node+1] < array[node]:
            count += 1
            node += 1
            node = int(node)
            if node >= len(array)-1:
                node = 0
    return count


def hilbert_trafo_nd(signal, axis=0):
    """simply the call of the off-shelf implementation to not have to calculate it for every feature individually.
    INPUT:
    :signal: (n,m)-dimensional array of real-valued signal. 
    We have activity=(rows,columns)=(time,nodes) -> default-axis=0.
    
    :output: (n,m)-dimensional array analytical representation of signal
    """
    from scipy.signal import hilbert
    
    #compute Hilbert Transform for analytical signal representation
    #ue.shape = (time-steps+1, number of nodes)
    #i.e. rows=time, columns=node -> want hilbert trafo w.r.t. time => axis=0
    ana_signal = hilbert(signal, axis=axis)
    
    return(ana_signal)

def hilbert_trafo_1d(signal):
    """simply the call of the off-shelf implementation to not have to calculate it for every feature individually"""
    from scipy.signal import hilbert
    
    ana_signal = hilbert(signal)
    
    return(ana_signal)

def inst_phase(signal):
    """Compute the instantaneous phase per time step per node."""
    
    #without unwrapping?
    inst_phase = np.unwrap(np.angle(signal))
    
    #inst_phase = np.unwrap(np.angle(signal))
    
    return inst_phase

def inst_frequ(signal):
    """ This function is supposed to determine the instantaneous frequency of a real-valued signal. 
    We use the method from Muller et al (2014), DOI: 10.1038/ncomms4675.
    
    INPUT:
    :signal: analytical representation (a+ib) of real-valued times series, 1-dimensional, array
    
    :output: instantaneous frequency without phase unwrapping, array"""
    
    
    #compute Hilbert Transform for analytical signal representation
    #ue.shape = (time-steps+1, number of nodes)
    #i.e. rows=time, columns=node -> want hilbert trafo w.r.t. time => axis=0
   # ana_signal = hilbert(signal)#, axis=0)
    
    complex_conj = np.conj(signal)
    #roll complex conjugate s.t. in product we compute 
    #(X_n\cdotX^*_{n+1} i.e. we multiply the analytical representation of signal 
    #at space step n with the complex conjugate of the next space step n+1)
    complex_conj = np.roll(complex_conj, -1)
    
    #use elementwise multiplication
    inst_frequ_temp = np.angle(np.multiply(signal, complex_conj))
    
    #omit last one, since it would be the product of X_n\cdotX^*_0
    inst_frequ = inst_frequ_temp[:-1]
    
    return inst_frequ
    

# explicit function to normalize array (for visualisation reasons - very helpful!)
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr




