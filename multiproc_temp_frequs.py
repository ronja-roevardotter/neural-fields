#IMPORTANT NOTE: This gathers the temporal frequencies of a single node
#THAT MEANS: It only works if all nodes show the same (periodic) acitivy (independet if time-shifted or not)

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
import pandas as pd

import matplotlib.pyplot as plt


from multiprocessing import Pool
from time import time
# importing datetime module for now()
import datetime

import py.continuum1d as continuum1d

from py.analysis import computeFPs, checkFixPtsStability

from py.funcs import getAvgPSD, getPSD
from py.params import setParams

def setNecVals():

    #set parameters to vary and DataFrame that's updated    
    params = {'b': 0.25, 'tau_a': 600, 'beta_a': 10, 'mu_a': 0.4, 'end_t': 30*1000}
    
    # creating new pandas DataFrame
    df_columns=['I_e', 'I_i', 'dom_frequ', 'down_duration', 'up_duration']
    df_temp = pd.DataFrame(columns=df_columns)
    
    return params, df_columns, df_temp

def collectFrequs(var1_val=0, var2_val=0):
    
    """
    This functions returns collected information on the fixed points:
    stability: type of stability. Options are: 0 (Hopf instability); 1 (1 stable fixed points); 2 (two stable fixed points)
    turing: checks the turing-stability for stable fixed points. 0 (all remain stable);  1 (at least one is turing unstable)
    p_random: emerging pattern, if randomly initialised
    p_turing: emerging pattern, if initialised in turing-unstable fixed point
    
    To have the origin of both varied params, transpose each matrix. 
    """
    
    
   # df_columns = df.columns.values.tolist()    

    params, df_columns, df_kill = setNecVals()
    
    params = setParams(params)
    
    params['I_e'] = var1_val
    params['I_i'] = var2_val
    
    fps = computeFPs(params)
    stab = checkFixPtsStability(fps, params)
    
    c1d = continuum1d.continuum1d()
    exc, inh = c1d.run(params, itype='inte_adaptation', fp = fps[0])
    
    duration = 10
    
    exc_test = exc.T[60,int(-duration*1000*(1/params.dt)):]
    
    frequs, PSD = getPSD(exc_test, int(1000*(1/params.dt)), nperseg=duration)

    
    dom_frequ = frequs[np.argmax(PSD)] 
    
    e = exc_test
    up_steps = sum(x>=0.4 for x in e)
    down_steps = sum(x<0.4 for x in e)
    if dom_frequ==0:
        avg_steps_in_down = 0
        avg_steps_in_up = 0
    else:
        avg_steps_in_down = down_steps/(duration*dom_frequ)
        avg_steps_in_up = up_steps/(duration*dom_frequ)
    
    avg_up = avg_steps_in_up*params.dt*(1/1000) #in [s]
    avg_down = avg_steps_in_down*params.dt*(1/1000) #in [s]
    
    values = [[var1_val, var2_val, dom_frequ, avg_down, avg_up]]
    
    return values


def run(var1, var2):
    
    vals_temp = collectFrequs(var1_val=var1, var2_val=var2)
    
    return vals_temp

def run_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return run(*a_b)

def pool_handler():
    
    df = pd.read_csv('csv/adaps-explo/high-adaps.csv')
    df_compute = df[df['p_down']==4]
    
    a = df_compute#df_show[df_show['p_random']==2]
    print(len(a))
    ext_values = np.zeros((len(a), 2))

    count = 0
    for index, row in a.iterrows():
       # print(row)
        ext_values[count][0] = row['I_e']
        ext_values[count][1] = row['I_i']
        count+=1
    
    # using now() to get current time
    current_time = datetime.datetime.now()
 
    # Printing value of now.
    print("Time now is:", current_time)
    
    start = time()
    print('start time: ', start)
    
    p = Pool(60)
    
    l = p.starmap(run, zip(ext_values[:,0], ext_values[:,1]))
    
    params, df_columns, df_kill = setNecVals()
    
    print('We are using default parameters with params-dict: %s' %str(params))
    
    # creating new pandas DataFrame
    df = pd.DataFrame(columns=df_columns)
    for ls in l:
        df_temp = pd.DataFrame(ls, columns=df_columns)
        df = pd.concat([df, df_temp])  #pd.DataFrame.from_records(l, columns=df_columns)

    filestr = 'high-adaps_temp_frequs.csv'

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

