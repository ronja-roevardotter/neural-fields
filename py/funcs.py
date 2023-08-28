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