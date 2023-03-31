#import necessary packages
import numpy as np

from py.integration import runIntegration
from py.params import setParams

import os

class continuum1d:
    
    def __init__(self):
        
        """
            
        This is the model class for the continuum Wilson-Cowan model, called mtype = 'activity'
        or else the continuum Amari model, called mtype = 'voltage'.
        System type: Integro-Differential equations with temporal and spatial component => dim(system)=2d
        IDEs for excitatory & inhibitory populations (coupled)
        Spatial dimension: dim(x)=1d, arranged on a ring 
        -> all-to-all same connectivity, determined by spread, omitting boundary conditions
            
        
        List of possible keys+values in params-dict:
        :mtype: determines the model type. Options: 'activity' for Wilson-Cowan type, 'voltage' for Amari type
        
        parameters:
        :w_ee: excitatory to excitatory coupling, float
        :w_ei: inhibitory to excitatory coupling, float
        :w_ie: excitatory to inhibitory coupling, float
        :w_ii: inhibitory to inhibitory coupling, float
        
        :tau_e: excitatory membrane time constant, float
        :tau_i: inhibitory membrane time constant, float
        
        :beta_e: excitatory gain (in sigmoidal transfer function), float
        :beta_i: inhibitory gain (in sigmoidal transfer function), float
        :mu_e: excitatory threshold (in sigmoidal transfer function), float
        :mu_i: inhibitory threshold (in sigmoidal transfer function), float
        
        :I_e: external input current to excitatory population, float
        :I_i: external input current to inhibitory population, float
        
        :kernel: what function used to determine spatial kernel, string, options are 'gaussian' or 'exponential'
        :sigma_e: characterizes the spatial extent of the excitatory to [...] connectivity, float
        :sigma_i: characterizes the spatial extent of the inhibitory to [...] connectivity, float
        
        temporal component:
        :dt: integration time step, float -> observe ms, therefore, setting e.g. dt=0.1, means we look at every 10th ms.
        :start_t: start of time intervall, integer
        :end_t: end of time intervall, integer
        
        spatial component:
        :n: number of pixels on ring, integer
        :length: length of total circumference of ring, float 
                 (remark: max. distance from pixel to furthest away can bi maximally length/2)
        :c: velocity of activity in [m/s], float -> transformed into mm/s in py.params.setParams()
        
        created by given params:
        :x: array of distances from one pixel to all other pixels (same distances to left and right, omit boundary effects), array
        :dx: spatial integration step, determined by length and n, float
        :ke: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
             determined by x, array
        :ki: kernel values (i.e. connectivity strengths) from excitatory population of a pixel to all other pixels, 
             determined by x, array
        :ke_fft: Fast Fourier Transform of ke by np.fft.fft, array
        :ki_fft: Fast Fourier Transform of ki by np.fft.fft, array
        
        :time: array of time intervall, array
        :delay: temporal delay from one pixel to another, determined by x,c and dt, array
        
        """
        print('1d Model initialised.')
        
        
    def run(self, params=None, fp=np.array([0.0, 0.01]), itype='inte_fft'):
        self.params = setParams(params)
        
        #itype determines the type of integration; string
        #options are: :inte_fft: integration by fourier transform, product, inverse fourier transform
                    # :inte_approxi: integral approximation of convolution
        return runIntegration(self.params, fp=fp, itype=itype)