import numpy as np
import matplotlib.pyplot as plt

import pandas as pd

import sys
import os

cdir = os.getcwd() #get current directory
os.chdir(cdir) #make sure you are in the current directory

#mother directory (i.e. one folder 'above')
mother_dir = os.path.abspath(os.path.join(cdir, os.pardir))
#verify  mother directory
print(f"Mother directory {mother_dir} given")

#get the path to the py directory
py_dir = os.path.abspath(os.path.join(os.getcwd(), '..', 'py'))
#add the py directory to the system path
if py_dir not in sys.path:
    sys.path.append(py_dir)

#verify the directory has been added
print(f"Added {py_dir} to sys.path")
print(f"System path: {sys.path}")

from opticalFlow import runOpticalFlow
from patternDetection import findContourPaths, contourMatrix, classifyCriticalPoints, avgNormVelocity, sequentialAvgNormVelocity
from spatiotemporalModes import concatenateReal, concatenateComplex, singularVectorDecomposition, proportionOfVariance, reshapeSpatialModes

#Import the 2-dimensional model
import twoD.continuum2d as continuum2d
c2d = continuum2d.continuum2d()

#import function to set parameters with the option to change specfic ones
from twoD.params2d import setParams

#import from twoD-folder the functions to determine the fixed points, their stability and whether they undergo a violation 
from twoD.analysis2d import computeFPs

def runProperties(data):

    df_columns=['time', 'u', 'v', 'order_param', 'critPoints_coords', 'patType']
    df = pd.DataFrame(columns=df_columns)       

    u, v, conv = runOpticalFlow(data, nSteps=50)

    X = u.shape[0]
    Y = u.shape[1]
    T = u.shape[2]

    for t in range(T):
        #collect order parameters for all t
        phi_t = avgNormVelocity(u.T[t], v.T[t])

        #collect critical points for all t
        u_conts = findContourPaths(u.T[t])
        v_conts = findContourPaths(v.T[t])  
        xcrossings = contourMatrix(u.T[t], u_conts)
        ycrossings = contourMatrix(v.T[t], v_conts)
        critcells = xcrossings * ycrossings
        coords = np.argwhere(critcells==1)

        #classify critical points
        rowcoords, colcoords, patType_t, jacs = classifyCriticalPoints(coords[:,0], coords[:,1], u.T[t], v.T[t])
    
        values = [[t, u.T[t], v.T[t], phi_t, coords, patType_t]]
        df_temp = pd.DataFrame(values, columns=df_columns)
        df = pd.concat([df, df_temp])

    return df, u, v


def runModes(u, v, real=True, amount_modes = 20):

    if real:
        w = concatenateReal(u.T, v.T)
    else: 
        w = concatenateComplex(u.T, v.T)

    
    R, T, Sigma = singularVectorDecomposition(w)
    vars = proportionOfVariance(Sigma)

    return R, vars

def runActivity(ext_input):
    ee = ext_input[0]
    ii = ext_input[1]

    params = {'I_e': ee, 'I_i': ii, 'n': 128, 'm': 128, 'end_t': 12*1000, 'pic_nmb': 48, 'xlength': 20, 'ylength': 20} 

    params = setParams(params)
    fps = computeFPs(params)

    ue, ui = c2d.run(params, itype='inte_adaptation', fp=fps[0])
    
    #Transform from dimension (T x X x Y) to dimension ()
    data = np.array(ue).T 
    data = data[:,:,8:] #cut of the transient time

    return data

def runall(ext_input, real=True, amount_modes=20):

    data = runActivity(ext_input)
    df, u, v = runProperties(data)

    amount_modes = u.shape[2]
    R, vars = runModes(u ,v, real, amount_modes)

    Rlist = [R[t] for t in range(amount_modes)]
    varslist = [vars[t] for t in range(amount_modes)]
   
    df['modes'] = Rlist
    df['vars'] = varslist

    #    df.iloc[t, -2] = R[t]
    #    df.iloc[t, -1] = vars[t]
    #    df[df['time']==t]['modes'] = R[t]
    #    df[df['time']==t]['variance_explained'] = vars[t]

    return df