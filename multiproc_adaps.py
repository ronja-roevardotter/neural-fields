# # # TO - DO # # # Update zu Adaps-Analyse!!! -> über git!

# change into the root directory of the project
import os
if os.getcwd().split("/")[-1] == "dev":
    os.chdir('..')
    
# get the current directory
cwd = os.getcwd()

# Print the current working directory
#print("Current working directory: {0}".format(cwd))
cwd_csv = cwd + '/csv/'


import numpy as np
import itertools

import pandas as pd
from multiprocessing import Pool
from time import time
# importing datetime module for now()
import datetime

from py.explo import collectPatterns
from py.analysis import setParams, computeFPs, checkFixPtsStability

from py.turings import checkStability


def setNecVals():

    #set parameters to vary and DataFrame that's updated    
    params = {'b': 0.25, 'tau_a': 600, 'beta_a': 10, 'mu_a': 0.4, 'end_t': 12*1000}
    
    var1_str = 'I_e'
    var2_str = 'I_i'
    
    # creating new pandas DataFrame
    df_columns = [var1_str, var2_str, 'stability', 'static', 'dynamic', 'double', 'p_random', 'p_down']
    df_temp = pd.DataFrame(columns=df_columns)
    
    return params, var1_str, var2_str, df_columns, df_temp

def collectStability(var1_val=0, var2_val=0):
    
    """
    This functions returns collected information on the fixed points:
    stability: type of stability. Options are: 0 (Hopf instability); 1 (1 stable fixed points); 2 (two stable fixed points)
    turing: checks the turing-stability for stable fixed points. 0 (all remain stable);  1 (at least one is turing unstable)
    p_random: emerging pattern, if randomly initialised
    p_turing: emerging pattern, if initialised in turing-unstable fixed point
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    
   # df_columns = df.columns.values.tolist()    

    params, var1_str, var2_str, df_columns, df_kill = setNecVals()
    
    ps = setParams(params)
    
    ps[var1_str] = var1_val
    ps[var2_str] = var2_val
    
    fps = computeFPs(ps)
    stab = checkFixPtsStability(fps, ps)
    
    static, dynamic, double = 0, 0, 0
    
    if sum(stab) == 2:
        stability = 2
        l=301
        k = np.linspace(0,2,l)
        static0, dynamic0, double0 = checkStability(k, fps[0], ps)
        static1, dynamic1, double1 = checkStability(k, fps[-1], ps)
        static, dynamic, double = max(static0, static1), max(dynamic0, dynamic1), max(double0, double1)
    elif sum(stab) == 1:
        stability = 1
        l=301
        k = np.linspace(0,2,l)
        static, dynamic, double = checkStability(k, fps[-1], ps)
    else:
        stability = 0
        
    
    p_random = collectPatterns(np.array([0.0, 0.01]), ps, last_sec=1) 
    
    if np.any(fps):
        p_down = collectPatterns(fps[0], ps, last_sec=1)  
    else:
        p_down = p_random
        stability = None
    
    
    values = [[var1_val, var2_val, stability, static, dynamic, double, p_random, p_down]]
    
    return values


def run(var1, var2):
    
    vals_temp = collectStability(var1_val=var1, var2_val=var2)
    
    return vals_temp

def run_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return run(*a_b)

def pool_handler():
    
    var1_vals = np.linspace(-1,1,201)
    var2_vals = np.linspace(-1,1,201)
    
    xx, yy = np.meshgrid(var1_vals, var2_vals)
    
    # using now() to get current time
    current_time = datetime.datetime.now()
 
    # Printing value of now.
    print("Time now is:", current_time)
    
    start = time()
    print('start time: ', start)
    
    p = Pool(70)
    
    l = p.starmap(run, zip(xx.flatten(), yy.flatten()))
    
    params, var1_str, var2_str, df_columns, df_kill = setNecVals()
    
    print('We are using default parameters with params-dict: %s' %str(params))
    print('Varied params are %s& %s' %(str(var1_str), str(var2_str)))
    
    # creating new pandas DataFrame
    df = pd.DataFrame(columns=df_columns)
    for ls in l:
        df_temp = pd.DataFrame(ls, columns=df_columns)
        df = pd.concat([df, df_temp])  #pd.DataFrame.from_records(l, columns=df_columns)

    filestr = 'high-adaps.csv'

    # writing empty DataFrame to the new csv file
    df.to_csv(cwd_csv + filestr)
    
    print('Simulation completed.')
    
    print('File is saved in %s%s' %(cwd_csv, filestr))
    runtime = time()-start
    print('RunTime: ', runtime)
    print(' ')
    minutes = runtime/60
    print('RunTime in minutes: ', minutes)
    print(' ')


if __name__=="__main__":
    pool_handler()










