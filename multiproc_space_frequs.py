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
    params = {'b': 0, 'tau_a': 600, 'beta_a': 10, 'mu_a': 0.4, 'end_t': 10*1000}
    
    # creating new pandas DataFrame
    df_columns=['I_e', 'I_i', 'bumps', 'down_space', 'up_space']
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
    
    exc_test = exc.T[:,-10000]
    
    exc_test = exc.T[:,-10000] #der integration-time-step der vorletzten Sekunde 
    #(IST NUR EIN STEP, FÜR TRAVELING WAVES: MUSS ÜBER ZEITRAUM GEHEN)
    
    frequs, PSD = getPSD(exc_test, params.n, nperseg=1)#params.n)
    
    dom_frequ = frequs[np.argmax(PSD)]    
    
    e = exc_test
    maxi = max(e)
    up_nodes = sum(x>=0.4*maxi for x in e)
    down_nodes = sum(x < 0.4*maxi for x in e)
    up_nodes, down_nodes
    if dom_frequ==0:
        nodes_per_down = 0
        nodes_per_up = 0
    else:
        nodes_per_down = down_nodes/(params.length*dom_frequ)
        nodes_per_up = up_nodes/(params.length*dom_frequ)
    space_per_down = params.dx * nodes_per_down
    space_per_up = params.dx * nodes_per_up
    
    values = [[var1_val, var2_val, dom_frequ, space_per_down, space_per_up]]
    
    return values


def run(var1, var2):
    
    vals_temp = collectFrequs(var1_val=var1, var2_val=var2)
    
    return vals_temp

def run_star(a_b):
    """Convert `f([1,2])` to `f(1,2)` call."""
    return run(*a_b)

def pool_handler():
    
    df = pd.read_csv('csv/default-explo/high-default.csv')
    df_compute = df[df['p_down']==3]
    
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

    filestr = 'high-default_temp_frequs.csv'

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

