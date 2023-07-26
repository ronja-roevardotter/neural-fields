# change into the root directory of the project
import os
if os.getcwd().split("/")[-1] == "dev":
    os.chdir('..')
    
# get the current directory
cwd = os.getcwd()

# Print the current working directory
print("Current working directory: {0}".format(cwd))
cwd_csv = cwd + '/csv/'


import numpy as np

import pandas as pd
from multiprocessing import Pool

from py.explo import collectPatterns

#set parameters to vary and DataFrame that's updated

vary1_str = 'I_e'
vary2_str = 'I_i'
    
vary1_vals = np.linspace(-1,1,4)
vary2_vals = np.linspace(-1,1,4)

# creating new pandas DataFrame
df_columns = [var1_str, var2_str, 'stability', 'turing', 'p_random', 'p_down', 'wavenumber']
df = pd.DataFrame(columns=df_columns)

print('We are using default parameters with params-dict: %s' %str(params))
print('Varied params are %s& %s' %(str(vary1_str), str(vary2_str)))



def collectStability(var1_val=0, var2_val=0, var1_str='I_e', var2_str='I_i', df, params=None,  pattern_analysis=False):
    
    """
    This functions returns collected information on the fixed points:
    stability: type of stability. Options are: 0 (Hopf instability); 1 (1 stable fixed points); 2 (two stable fixed points)
    turing: checks the turing-stability for stable fixed points. 0 (all remain stable);  1 (at least one is turing unstable)
    p_random: emerging pattern, if randomly initialised
    p_turing: emerging pattern, if initialised in turing-unstable fixed point
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    
    df_columns = df.columns.values.tolist()
    
    params[var1_str] = var1_val
    params[var2_str] = var2_val
    
    ps = setParams(params)
    fps = computeFPs(ps)
    stab = checkFixPtsStability(fps, ps)
    
    violation = 0
    k0 = None
    
    if sum(stab) == 2:
        stability = 2
        l=101
        k = np.linspace(-2,2,l)
        a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[0], ps)
        vio1, k0 = violationType(k, a_ee, a_ei, a_ie, a_ii, ps)
        a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[-1], ps)
        vio2, k02 = violationType(k, a_ee, a_ei, a_ie, a_ii, ps)
        violation = max(vio1, vio2)
    elif sum(stab) == 1:
        stability = 1
        l=101
        k = np.linspace(-2,2,l)
        a_ee, a_ei, a_ie, a_ii = a_jkValues(fps[list(stab).index(1)], ps)
        violation, k0 = violationType(k, a_ee, a_ei, a_ie, a_ii, ps)
    else:
        stability = 0
        
    
    p_random = collectPatterns(np.array([0.0, 0.01]), ps, last_sec=1) 
    
    if np.any(fps):
        p_down = collectPatterns(fps[0], ps, last_sec=1)  
    else:
        p_down = p_random
        stability = None
    
    
    values = [[var1_val, var2_val, stability, violation, p_random, p_down, k0]]#, p_turing]]
    df_temp = pd.DataFrame(values, columns=df_columns)
    df = pd.concat([df, df_temp])
    
#    return df


def run(vary1, vary2):
    
    params = {'I_e': 0, 'I_i': 0, 'b': 0.5, 'tau_a': 1000, 'end_t': 10*1000}
    
    collectStability(var1_val=vary1, var2_val=vary2, var1_str=var1_str, var2_str=var2_str, df, 
                     params=None,  pattern_analysis=False)

    
    return df



if __name__="__main__":
    
    p = Pool(2)
    
    p.map(run, zip(vary1_vals, vary2_vals))

    filestr = 'trial.csv'

    # writing empty DataFrame to the new csv file
    df.to_csv(cwd_csv + filestr)
    
    print('Simulation completed.')
    
    print('File is saved in %s%s' %(cwd_csv, filestr))










