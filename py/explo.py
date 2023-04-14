import numpy as np
import pandas as pd

from py.params import setParams
from py.analysis import computeFPs, checkFixPtsStability, a_jkValues
from py.analysis import tr, det, lmbd

from py.funcs import getAvgPSD

from py.kernels import gaussian, exponential

import py.continuum1d as continuum1d




# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # naive Ansatz to identify whether a homogeneous steady state remains stable in the presence of spatial interaction # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #

def checkTuringStability(det, tr):
    if all(det>=0) and all(tr<=0):
        turing=0
    else:
        turing=1

    return turing


# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # - - - - - - - naive Ansatz to collect information about stability, turing stability and pattern type - - - - - - - # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #


def collectStabilities(params=None, vary_params={'I_e': np.linspace(1,5,21), 'I_i': np.linspace(0,4,21)}, pattern_analysis=False):
    
    """
    This functions returns three matrices. One shows the fixed points and their stability for a single node (no spatial connectivity to others). 
    The second shows turing instability for the nodes that were identified as stable in matrix 1. 
    The third schows the underlying activity patterns. -1 if not identifiable, 
    1: homogeneous steady state, 
    2: temporal
    3: spatial
    4: spatiotemporal, and
    5: multiple fixed points lead to different patterns.
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    var1_str, var1 = list(vary_params.items())[0]
    var2_str, var2 = list(vary_params.items())[1]
 #   var2 = var2[::-1]
    
    print(type(var1_str), type(var1))
    
    df_columns=[var1_str, var2_str, 'fixed_points', 'stability', 'turing_stabilities', 'patterns']
    df_columns=[var1_str, var2_str, 'fp1', 'stab1', 't_stab1', 'p1', 
                                    'fp2', 'stab2', 't_stab2', 'p2', 
                                    'fp3', 'stab3', 't_stab3', 'p3']
    df = pd.DataFrame(columns=df_columns)
    
    nn = len(var1)
    mm = len(var2)
    
    mass_bifs = np.zeros((nn,mm))
    pattern_mtx = np.zeros((nn,mm))
    
    for i in range(nn):
        for j in range(mm):
            params[var1_str] = var1[i]
            params[var2_str] = var2[j]
            
            ps = setParams(params)
            fps = computeFPs(ps)
            fps_for_later = fps
            
            stab = checkFixPtsStability(fps, ps)
            stab_for_later = stab
            
            if len(fps)==1:
                if fps[0][0] <=0.4:
                    fps = np.array([fps[0], None, None])
                    stab = np.array([stab[0], None, None])
                    trng_stab = np.array([0, None, None])
                    patterns = np.array([0, None, None])
                else:
                    fps = np.array([None, None, fps[0]])
                    stab = np.array([None, None, stab[0]])
                    trng_stab = np.array([None, None, 0])
                    patterns = np.array([None, None, 0])
            elif len(fps)==2:
                fps = np.array([fps[0], None, fps[-1]])
                stab = np.array([stab[0], None, stab[-1]])
                trng_stab = np.array([0, None, 0])
                patterns = np.array([0, None, 0])
            else:
                trng_stab = np.array([0, 0, 0])
                patterns = np.array([0, 0, 0])
            
            
            if sum(stab_for_later) == 2:
                mass_bifs[i,j] = 1
                l=61
                k = np.linspace(-2,2,l)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps_for_later[0], ps)
                determinant1 = det(k, a_ee, a_ei, a_ie, a_ii, ps)
                trace1 = tr(k, a_ee, a_ii, ps)
                trng_stab[0] = checkTuringStability(determinant1, trace1)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps_for_later[-1], ps)
                determinant2 = det(k, a_ee, a_ei, a_ie, a_ii, ps)
                trace2 = tr(k, a_ee, a_ii, ps)
                trng_stab[-1] = checkTuringStability(determinant2, trace2)
            elif sum(stab_for_later) == 1:
                mass_bifs[i,j] = 0.5
                l=61
                k = np.linspace(-2,2,l)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps_for_later[list(stab_for_later).index(1)], ps)
                determinant = det(k, a_ee, a_ei, a_ie, a_ii, ps)
                trace = tr(k, a_ee, a_ii, ps)
                trng_stab[list(stab).index(1)] = checkTuringStability(determinant, trace)
            else:
                mass_bifs[i,j] = 0
                
            
            for idx, fp in enumerate(fps_for_later):
                p, a,b,c,d = collectPatterns(fp, ps, last_sec=100)
                patterns[idx] = p
            
            
            values = [[var1[i], var2[j], fps[0], stab[0], trng_stab[0], patterns[0],
                                         fps[1], stab[1], trng_stab[1], patterns[1],
                                         fps[2], stab[2], trng_stab[2], patterns[2]]]
            df_temp = pd.DataFrame(values, columns=df_columns)
            df = pd.concat([df, df_temp])
                
            
      #      print('We are in round I_e = %f, I_i = %f, i=%i, j=%i with pattern %s' %(var1[i],var2[j],i,j, str(patterns)))
    
    return mass_bifs, df



# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # - - - - - - - - - - - - - - - - - - - - naive Ansatz to identify pattern type - - - - - - - - - - - - - - - - - - # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #


def collectPatterns(fp, params, last_sec=100):
    
    """ This function collects the type of activity-pattern that is shown after running a simulation for different settings of parameters 
    (fix given by params, varied in trng-df DataFrame) initialized in each available fixed point per parametrization. 
    Pattern-Identification on basis of frequency over space and over time.
    
    INPUT:
    :mtype: type of model, string, either 'activity' or 'voltage' 
    :trng_df: pandas-DataFrame, column 1 and column 2 in trng_df are the parameters that were varied, 
              there is one column that gives all available fixed points per parametrization, named 'fixed_points' 
    :params: dictionary of fix parameters
    :variables: dictionary of variables (duration, #pixels,  delta_t)
    
    OUTPUT:
    :df: pandas-Dataframe with [varied_param1, varied_param2, patterns], where 'patterns' is a list of the same length as 'fixed_points' with numbers from 1-4, 
    indicating the emerging pattern after initialising the model in the corresponding fixed point.
    stationary=1
    temporal=2
    spatial=3
    spatiotemporal=4
    e.g. parametrization shows 3 fixed points, [fp1, fp2, fp3], init in fp1 shows spatial, in fp2 &fp3 stationary patterns => patterns=[3,1,1]"""
    
    c1d = continuum1d.continuum1d()
    
    exc, inh = c1d.run(params, itype='inte_fft', fp=fp)
        
   # print('exc[-10]', exc[-10])
   # print('exc.T[-10]', exc.T[-10])
    #the returned activity is returned in shape: rows per time step, len(row)=#of pixels
    #we transpose that to have a matrix with one row per pixel, and coulmns=time steps.
    x = exc.T
    temp = int(last_sec*(1/params.dt))
    x = x[:,-temp:]
    
 #   print('x before Pxx computation', x.flatten() )
    
    
    #to identify whether there is change over time per node 
    #("per node there is a frequency>0 => temporal pattern"), we get the average
    #PSD over time and check, whether all(power(frequencies)) are close to 0: 
    #if returned false, then there is a frequency with power >0
    #i.e. a change in activity over time => temporal pattern
    #investigated time series-matrix: x=(rows=nodes, columns=time steps), 
    #e.g. 37 nodes and 5 seconds => shape=(37,5*variables['delta_t'])
    
    fs = params.n
    f_time, Pxx_den_time = getAvgPSD(x, fs)
    temporally_homogeneous = all(Pxx_den_time <=0.1*(10**(-5)))  #all(np.isclose(Pxx_den_time,0))
    
    #to identify vise verca, if there is a change in activiy over space, 
    #we check the frequency over nodes per time step, 
    #hence transpose x again
    x=x.T
    fs = params.dt
    f_space, Pxx_den_spatial = getAvgPSD(x, fs)
    spatially_homogeneous = all(Pxx_den_spatial <=0.1*(10**(-5)))  #np.isclose(Pxx_den_spatial,0))
    
    if spatially_homogeneous and temporally_homogeneous:
        pattern = 1
    elif spatially_homogeneous and not temporally_homogeneous:
        pattern = 2
    elif not spatially_homogeneous and temporally_homogeneous:
        pattern = 3
    else:
        pattern = 4
        
 #   print('In pattern collection, pattern: ', pattern)
        
    return pattern, Pxx_den_time, f_time, Pxx_den_spatial, f_space


