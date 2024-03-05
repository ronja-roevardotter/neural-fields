#This py-file contains the 'run functions', i.e. the functions used to run certain pipelines, e.g. the pipeline for phase latencies
import numpy as np

import sys
sys.path.append('/Users/ronja/Documents/GitHub/neural-fields/py')

from funcs import count_nodes_for_descent, getSwitchIndex, hilbert_trafo_nd, inst_phase, rotation_in_latency
from params import setParams
from analysis import a_jkValues, violationType, checkFixPtsStability, computeFPs
from turings import checkStability

import continuum1d as continuum1d
c1d = continuum1d.continuum1d()


def run_violation(params, fp):
    """
    This functions determines whether the fixed point 'fp' undergoes an instability under consideration of a spatial perturbation
    INPUT:
    :params: dicitionary of parameters to configure the model
    :fp: 2-entry array of [fixed_point(u_e), fixed_point(ui)]

    OUPUT:
    :vio: case b==0: integer, either 0 (no violation), 1 (static Turing), 2 (dynamic Turing), 3 (both)
          case b!=0: three entries, each indicating if it undergoes (static Turing, dynamic Turing, Takens-Bogdanov) of instability (1) or not (0)
    """

    #you have to call for different possible violations to undergoe instability
    #depends whether we have a 2-dim dynamical system (e.g. only exc-inh), or a 3-dim dynamical system (e.g. exc-inh-adaps)
    params=setParams(params)
    k = np.linspace(0,3,201)

    if params.b == 0:
        a_ee, a_ei, a_ie, a_ii = a_jkValues(fp, params)
        vio = violationType(k, a_ee, a_ei, a_ie, a_ii, params) 
    else:
        vio = checkStability(k, fp, params)

    return vio

def run_fixedpoints(params):
    """"
    This functions computes the fixed points, their mass stability (i.e. without spatial spread) for the parameter-dictionary
    INPUT:
    :params: dicitionary of parameters to configure the model

    OUTPUT:
    :fps: np-array of fixed point values in exc,inh-pairs 
    :stab: array of length=len(fps), each entry indicating whether the fixed point at corresponding position in fps is stable (1) or not (0)    
    """

    params = setParams(params)
    fps = computeFPs(params)
    stab = checkFixPtsStability(fps, params)

    return fps, stab


def run_activity(params, itype='integrate_conv', fp=np.array([0.0, 0.01])):
    """
    This functions simulates the integration over an excitatory and inhibitory population of neurons in the 1d Wilson-Cowan Continuum.
    Setting is given in the params-dictionary

    INPUT:
    :params: dicitionary of parameters to configure the model
    :itype: string that identifies the method of integration. Options are: 'integrate_conv' and 'integrate_approxi'
    :array: 2-entry array [fixed_point(u_e), fixed_point(ui)] that indicates, whether the activities are initialised in the fp.

    OUPUT:
    :ue, ui: time x nodes-dimensional array of the activity of excitatory (ue) and inhibitory (ui) populations
    """

    params = setParams(params)    
    ue, ui = c1d.run(params, itype=itype, fp=fp)
    
    return ue, ui


def run_latencies(params):

    """
     This function runs the entire computation to get the phase latency of a periodic traveling wave in the spatially 1d-model. 
     INPUT:
     :params: The parameters f your choice for which you want to simulate the activity.

     OUTPUT: (here, only for the excitatory activity)
     :amount_time_steps: a  integer that represents the counted amount of time steps it takes the activity to travel one full-cycle oscillation
     :amount_of_nodes: an integer that represents the amound of nodes (i.e. the space)
    """
    
    params = setParams(params)
    fps, stab = run_fixedpoints(params)
    fp=fps[0]

    vio = run_violation(params, fp)

    print('for I_e=%.2f, I_i=%.2f' %(params.I_e,params.I_i))
    print('fixed points %s ' %str(fps))
    print('with (mass) stability %s'  %str(stab))
    print('and continuum stability of down: %s' %str(vio))
    
    ue, ui = run_activity(params, fp=fp)
    
    duration = 10
    dur_steps = int(duration * (1/ps.dt) * 1000)
    signal = hilbert_trafo_nd(ue[-dur_steps:,:], axis=0)
    phases = inst_phase(signal)
    
    phases_cut = phases[-80000:]
    
    phase_latencies = np.zeros(len(phases_cut.T))
    
    for idx, node in enumerate(phases_cut.T):
        complex_vector = np.exp(1j * node)
        
        # Extract real and imaginary components of the complex vectors
        real_part = np.real(complex_vector)
        imaginary_part = np.imag(complex_vector)
        
        how_many_time_steps = getSwitchIndex(imaginary_part)
        #print(how_many_time_steps[0])
        if len(how_many_time_steps)<=1:
            phase_latencies[idx] = 0
        else:
            #take the second switch to count the #time steps for a full cycle
            phase_latencies[idx] = how_many_time_steps[1]
        
    rotation = rotation_in_latency(phase_latencies)
    amount_of_nodes = count_nodes_for_descent(phase_latencies, rotation)
    amount_time_steps = phase_latencies[np.argmax(phase_latencies)]
    
    print(r'$I_e=$' + '%.2f' %params.I_e + r'$\ and\ I_i=$' + '%.2f' %params.I_i)
    print('time steps: ', amount_time_steps)
    print('node steps: ', amount_of_nodes)
        
    return amount_time_steps, amount_of_nodes

def run_latencies_for_activity(params, ue):

    """
     This function runs the entire computation to get the phase latency of a periodic traveling wave in the spatially 1d-model. 
     INPUT:
     :params: The parameters f your choice for which you want to simulate the activity.

     OUTPUT: (here, only for the excitatory activity)
     :amount_time_steps: a  integer that represents the counted amount of time steps it takes the activity to travel one full-cycle oscillation
     :amount_of_nodes: an integer that represents the amound of nodes (i.e. the space)
    """
    
    params = setParams(params)
    
    duration = 10
    dur_steps = int(duration * (1/params.dt) * 1000)
    signal = hilbert_trafo_nd(ue[-dur_steps:,:], axis=0)
    phases = inst_phase(signal)
    
    phases_cut = phases[-80000:]
    
    phase_latencies = np.zeros(len(phases_cut.T))
    
    for idx, node in enumerate(phases_cut.T):
        complex_vector = np.exp(1j * node)
        
        # Extract real and imaginary components of the complex vectors
        real_part = np.real(complex_vector)
        imaginary_part = np.imag(complex_vector)
        
        how_many_time_steps = getSwitchIndex(imaginary_part)
        #print(how_many_time_steps[0])
        if len(how_many_time_steps)<=1:
            phase_latencies[idx] = 0
        else:
            #take the second switch to count the #time steps for a full cycle
            phase_latencies[idx] = how_many_time_steps[1]
        
    rotation = rotation_in_latency(phase_latencies)
    amount_of_nodes = count_nodes_for_descent(phase_latencies, rotation)
    amount_time_steps = phase_latencies[np.argmax(phase_latencies)]
    
    print(r'$I_e=$' + '%.2f' %params.I_e + r'$\ and\ I_i=$' + '%.2f' %params.I_i)
    print('time steps: ', amount_time_steps)
    print('node steps: ', amount_of_nodes)
        
    return amount_time_steps, amount_of_nodes