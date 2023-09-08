import numpy as np
import pandas as pd

from py.twoD.params2d import setParams
from py.twoD.analysis2d import computeFPs, checkFixPtsStability, a_jkValues, violationType
from py.twoD.analysis2d import tr, det, lmbd

from py.twoD.turings2d import checkStability

from py.funcs import getAvgPSD

from py.twoD.kernels2d import gaussian

import py.twoD.continuum2d as continuum2d

c2d = continuum2d.continuum2d()




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
                    fps = np.array([fps[0], None, None], dtype='object')
                    stab = np.array([stab[0], None, None], dtype='object')
                    trng_stab = np.array([0, None, None], dtype='object')
                    patterns = np.array([0, None, None], dtype='object')
                else:
                    fps = np.array([None, None, fps[0]], dtype='object')
                    stab = np.array([None, None, stab[0]], dtype='object')
                    trng_stab = np.array([None, None, 0], dtype='object')
                    patterns = np.array([None, None, 0], dtype='object')
            elif len(fps)==2:
                fps = np.array([fps[0], None, fps[-1]], dtype='object')
                stab = np.array([stab[0], None, stab[-1]], dtype='object')
                trng_stab = np.array([0, None, 0], dtype='object')
                patterns = np.array([0, None, 0], dtype='object')
            elif len(fps)==3:
                trng_stab = np.array([0, 0, 0])
                patterns = np.array([0, 0, 0])
            else:
                fps = np.array([None, None, None], dtype='object')
                stab = np.array([None, None, None], dtype='object')
                trng_stab = np.array([None, None, None], dtype='object')
                patterns = np.array([None, None, None], dtype='object')
                
            
            
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
                p, a,b,c,d = collectPatterns(fp, ps, last_sec=1)
                patterns[idx] = p
            
            
            values = [[var1[i], var2[j], fps[0], stab[0], trng_stab[0], patterns[0],
                                         fps[1], stab[1], trng_stab[1], patterns[1],
                                         fps[2], stab[2], trng_stab[2], patterns[2]]]
            df_temp = pd.DataFrame(values, columns=df_columns)
            df = pd.concat([df, df_temp])
                
            
        print('We finished round I_e = %f, i=%i' %(var1[i],i))
    print('df: ', df)
    
    return mass_bifs, df



# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #
# # # - - - - - - - - - - - - - - - - - - - - naive Ansatz to identify pattern type - - - - - - - - - - - - - - - - - - # # #
# # # # # # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # # # # # # # # #


def collectPatterns(fp, params):
    
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
    
#    c2d = continuum1d.continuum1d()

    if params.b==0:
        itype = 'inte_fft'
    else:
        itype = 'inte_adaptation'
        
    
    exc, inh = c2d.run(params, itype=itype, fp=fp)
        
    #the returned activity is returned in shape: rows per time step, len(row)=#of pixels (i.e. = #columns)
    #we transpose that to have a matrix with one row per pixel, and coulmns=time steps.
    x = exc[-1].T
    frequs = np.fft.fftshift(np.fft.fft2(x))
    
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    
    norm = sigmoid(frequs)
    reals = sigmoid(frequs.real)
    imags = sigmoid(frequs.imag)
    
    abs_diff = np.max(np.log(abs(norm))) - np.min(np.log(abs(norm)))
    real_diff = np.max(reals) - np.min(reals)
    imag_diff = np.max(imags) - np.min(imags)
    
    pattern = 0
    #stationary or temporal (no change over space)
    if np.isclose(real_diff, 0.5, atol=0.1e-5) and imag_diff<0.1e-9:
        #check further, it it changes over time or not
        pattern = 1
    elif real_diff>0.8 and not imag_diff>0.8:
        #check further, it it changes over time or not
        pattern = 2
    else:
        pattern = 0
        
 #   print('In pattern collection, pattern: ', pattern)
        
    return pattern #, Pxx_den_time, f_time, Pxx_den_spatial, f_space


def collectStabilities2(params=None, vary_params={'I_e': np.linspace(1,5,21), 'I_i': np.linspace(0,4,21)}, pattern_analysis=False):
    
    """
    This functions returns collected information on the fixed points:
    stability: type of stability. Options are: 0 (Hopf instability); 1 (1 stable fixed points); 2 (two stable fixed points)
    turing: checks the turing-stability for stable fixed points. 0 (all remain stable);  1 (at least one is turing unstable)
    p_random: emerging pattern, if randomly initialised
    p_turing: emerging pattern, if initialised in turing-unstable fixed point
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    var1_str, var1 = list(vary_params.items())[0]
    var2_str, var2 = list(vary_params.items())[1]
 #   var2 = var2[::-1]
    
    print(type(var1_str), type(var1))
    df_columns=[var1_str, var2_str, 'stability', 'turing', 'p_random', 'p_down', 'wavenumber']
    df = pd.DataFrame(columns=df_columns)
    
    nn = len(var1)
    mm = len(var2)
    print('nn=%i, mm=%i' %(nn,mm))
    
    for aa in range(0,nn):
        for bb in range(0,mm):
         #   print('we are in round (aa,bb)= (%i, %i)' %(aa,bb))
            params[var1_str] = var1[aa]
            params[var2_str] = var2[bb]
            
            ps = setParams(params)
        #    print('parameters set: ', str(ps))
            fps = computeFPs(ps)
            stab = checkFixPtsStability(fps, ps)
            
            violation = 0
            k0 = None
            
            if sum(stab) == 2:
                stability = 2
                l=101
                k = np.linspace(0,2,l)
                kk = [] #np.zeros(int((((N-1)*N)/2)+N))

                for idx1 in range(l):
                    for idx2 in range(idx1, l):
                        kk.append(k[idx1]**2+k[idx2]**2)
                kk.sort()
                kk = np.array(kk)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[0], ps)
                vio1, k0 = violationType(kk, a_ee, a_ei, a_ie, a_ii, ps)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[-1], ps)
                vio2, k02 = violationType(kk, a_ee, a_ei, a_ie, a_ii, ps)
                violation = max(vio1, vio2)
                
            elif sum(stab) == 1:
                stability = 1
                l=101
                k = np.linspace(0,2,l)
                kk = [] #np.zeros(int((((N-1)*N)/2)+N))

                for idx1 in range(l):
                    for idx2 in range(idx1, l):
                        kk.append(k[idx1]**2+k[idx2]**2)
                kk.sort()
                kk = np.array(kk)
                a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[list(stab).index(1)], ps)
                violation, k0 = violationType(kk, a_ee, a_ei, a_ie, a_ii, ps)
                
            else:
                stability = 0
                
            
            p_random = collectPatterns(np.array([0.0, 0.01]), ps) 
            p_down = 0
            if np.any(fps):
                lueckenfueller=1
                p_down = collectPatterns(fps[0], ps)  
            else:
                p_down = p_random
                stability = None
            
            
            values = [[var1[aa], var2[bb], stability, violation, p_random, p_down, k0]]#, p_turing]]
            df_temp = pd.DataFrame(values, columns=df_columns)
            df = pd.concat([df, df_temp])
                
            
            print('We finished round I_e = %f, I_i=%f, i=%i, j=%i' %(var1[aa],var2[bb],aa,bb))
 #   print('df: ', df)
    
    return df


def collectAdapStabs(params=None, 
                     vary_params={'I_e': np.linspace(1,5,21), 'I_i': np.linspace(0,4,21)}, 
                     pattern_analysis=False):
    
    """
    This functions returns collected information on the fixed points:
    stability: type of stability. Options are: 0 (Hopf instability); 1 (1 stable fixed points); 2 (two stable fixed points)
    turing: checks the turing-stability for stable fixed points. 0 (all remain stable);  1 (at least one is turing unstable)
    p_random: emerging pattern, if randomly initialised
    p_turing: emerging pattern, if initialised in turing-unstable fixed point
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    var1_str, var1 = list(vary_params.items())[0]
    var2_str, var2 = list(vary_params.items())[1]
 #   var2 = var2[::-1]
    
    print(type(var1_str), type(var1))
    df_columns=[var1_str, var2_str, 'stability', 'turing', 'p_random', 'p_down']
    df = pd.DataFrame(columns=df_columns)
    
    nn = len(var1)
    mm = len(var2)
    
    for i in range(nn):
        for j in range(mm):
            params[var1_str] = var1[i]
            params[var2_str] = var2[j]
            
            ps = setParams(params)
            fps = computeFPs(ps)
            stab = checkFixPtsStability(fps, ps)
            
            turing = 0
            
            if sum(stab) == 2:
                stability = 2
                l=61
                k = np.linspace(0,3,l)
                turing0 = checkStability(k, fps[0], ps)
                turing1 = checkStability(k, fps[-1], ps)
                turing = max(turing0, turing1)
            elif sum(stab) == 1:
                stability = 1
                l=101
                k = np.linspace(-2,2,l)
                turing = checkStability(k, fps[-1], ps)
            else:
                stability = 0
                
            
            p_random = collectPatterns(np.array([0.0, 0.01]), ps, last_sec=1) 
            if np.any(fps):
                p_down = collectPatterns(fps[0], ps, last_sec=1)  
            else:
                p_down = p_random
                stability = None
            
            
            values = [[var1[i], var2[j], stability, turing, p_random, p_down]]#, p_turing]]
            df_temp = pd.DataFrame(values, columns=df_columns)
            df = pd.concat([df, df_temp])
                
            
        print('We finished round I_e = %f, i=%i' %(var1[i],i))
 #   print('df: ', df)
    
    return df



