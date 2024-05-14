import numpy as np
from scipy.sparse import diags, linalg
import logging

def penalty(x, beta):
    return np.sqrt(x**2 + beta**2)

def penaltyPrime(x, beta):
    return (2*np.sqrt(x**2 + beta**2))**(-1)

def opticalFlowStep_old(array, alpha, beta, Dx, Dy, Dt, nSteps, params):
    """
    This function computes the optical flow into x- and y-direction for a given data matrix (e.g. one time step in a series)

    INPUT:
    :array: (X x Y) dimensional array, not needed
    :nSteps: integer of maximal amount of iteration steps for fixed point iteration

    OUTPUT:
    :u:
    :v: (X x Y) dimensional array
    :nConv: integer, amount of steps until 'convergence'. If nConv > nSteps, the fp-iteration hasn't converged.
    """
    #set annealing parameter (also called relaxation parameter) for fixed-point iteration
    relaxParam = 1.1
    relaxStep = 0.02
    relaxParamMin = 0.1

    #initialise parameters
    u = np.zeros((array.shape))
    v = np.zeros((array.shape))
    diff = 20
    nConv = 0

    Ed = np.full_like(array, np.inf)
    Es = np.full_like(array, np.inf)

    while diff >= 0.1e-5 and nConv < nSteps:

        #remember old values to check convergence later
        Ed_old = Ed
        Es_old = Es
        u_old = u
        v_old = v

        #compute new Ed, Es, Laplacians of u, v 
        Ed = Dx * u + Dy * v + Dt
        uLap = spatialLaplacian(u, params)
        vLap = spatialLaplacian(v, params)
        Es = abs(uLap)**2 + abs(vLap)**2 # already squared because we have no definition for Es non-squared, except for taking the root which is unnecessary here.

        #compute the penalty pd, ps and their prime
        # not the prime: pd = (1/alpha) * penalty(Ed**2, beta)
        # not the prime: ps = penalty(Es, beta)
        pd = (1/alpha) * penaltyPrime(Ed**2, beta)
        ps = penaltyPrime(Es, beta)
        delta = np.zeros(Ed.shape).astype(float)

        #compute the distance between the new Ed and old Ed (Es analogously) to check whether it has converged early
        diffEd = abs(Ed_old - Ed) / abs(Ed)
        diffEs = abs(Es_old - Es) / abs(Es) 
        if np.max(diffEs) < 0.1e-6 and np.max(diffEd) < 0.1e-6:
            break

        #reformulate equations s.t. we have a linear system of the Form Ax=b
        #doing this fully analogue to Townsend and Gong (2018).

        #print('All parameters inlcuding the convergence-check were done in opticalFlowStep.')
       
        #set parameters X, Y, T, initialise vx & vy
        xdim = array.shape[0]
        ydim = array.shape[1]
        N = xdim * ydim

        #unsure about this: the array 'surroundTerms' is a flattened array of length N of values at positions 
        #of 'surrounding locations' i.e. the locations used to determine Dx, Dy, Dt with 5-point-stencil
        #the values are the derivatives of ps-to-x ps-to-y and ps-to-laplacian
        #I can not translate this literally since I have a torus and don't have any 'positions calcluation',
        #neither any neighbor determination.
        #Therefore, I will make surroundTerms a xdim x ydim dimensional matrix, with values of psx, psy, pslap around the center
        #for each 'dx'/'dy' distance used respectively
        psx, psy = spatialPartialDerivative(ps, params)
        pslap = spatialLaplacian(ps, params)

       # print(psx, psx.shape)

        surroundTerms = np.zeros((xdim, ydim))

        #find locations in matrix that correspond to steps in determination of spatial partial derivatives
        center = [int(xdim/2), int(ydim/2)]
        xdir_locs = [[center[0]-2, center[1]], 
                        [center[0]-1, center[1]], 
                        [center[0], center[1]],
                        [center[0]+1, center[1]],
                        [center[0]+2, center[1]]]
        ydir_locs = [[center[0], center[1]-2], 
                        [center[0], center[1]-1], 
                        [center[0], center[1]],
                        [center[0], center[1]+1],
                        [center[0], center[1]+2]]
        lap_locs = [[center[0]-1, center[1]], 
                        [center[0]+1, center[1]], 
                        [center[0], center[1]],
                        [center[0], center[1]-1],
                        [center[0], center[1]+1]]
        
        xlocs = np.zeros((xdim, ydim))
        ylocs = np.zeros((xdim, ydim))
        laplocs = np.zeros((xdim, ydim))
        # Set values at specified positions to 1
       # for row, col in xdir_locs:
       #     xlocs[row, col] = psx
        
       # for row, col in xdir_locs:
       #     ylocs[row, col] = psy
        
       # for row, col in xdir_locs:
       #     laplocs[row, col] = pslap

        surroundTerms += psx + psy + pslap

       # print(surroundTerms, surroundTerms.shape, surroundTerms.ravel(), surroundTerms.ravel().shape)

        # # # - - - The following code is a literal translation from Townsend & Gongs matlab code, provided by ChatGPT - - - # # #
        # Calculate b vector
        b = np.hstack((pd.ravel() * Dt.ravel() * Dx.ravel(), pd.ravel() * Dt.ravel() * Dy.ravel()))

        # Add diagonal terms
        diag_vals = np.hstack((-delta.ravel() - Dx.ravel()**2 * pd.ravel(), -delta.ravel() - Dy.ravel()**2 * pd.ravel()))

        # Create sparse diagonal matrix
        A = diags(diag_vals, format='csc')

        #print('b & A were successfully defined.')

        # Add off-diagonal terms for ui-vi dependence
        uvDiag = -Dx.ravel() * Dy.ravel() * pd.ravel()
        p_off_diag = diags([uvDiag, uvDiag], [N, -N], format='csc', shape=(2 * N, 2 * N))
        A = A + p_off_diag

        # Add other terms for surrounding locations
        #sp_zeros = diags([0], [0], shape=(N, N), format='csc')
        #print(sp_zeros)
        surroundTerms = diags([surroundTerms.ravel(), surroundTerms.ravel()], [N, -N], shape=(2 * N, 2 * N), format='csc')
        A = A + surroundTerms #diags([surroundTerms.ravel(), np.zeros((2*N))], [N, -N], shape=(2 * N, 2 * N), format='csc')

        #print('surroundTerms and A were succesfully updated.')

        # Solve this system of linear equations, adding a small value along the diagonal
        # to avoid potentially having a singular matrix
        diag_small_value = diags([1e-10], [0], shape=(2 * N, 2 * N), format='csc')
        A = A + diag_small_value

        #print('diag_small_value and final update on A were successfully done.')

        xexact = linalg.spsolve(A.toarray(), b)

        #print('xexact (which is solving A/b) has sucessfully been computed')

        # Reshape back to grids
        u = (1 - relaxParam) * u + relaxParam * np.reshape(xexact[:N], (xdim, ydim))
        v = (1 - relaxParam) * v + relaxParam * np.reshape(xexact[N:], (xdim, ydim))

        # Gradually reduce the relaxation parameter to ensure the fixed point
        # iteration converges
        if relaxParam > relaxParamMin:
            relaxParam = relaxParam - relaxStep

        nConv +=1

        if nConv%10==0:
            print('inside step: VVF at time step %i is calculated' %nConv)


        # # # - - YOU STOPPED IN THE MIDDLE OF THE FP-ITERATION so HERE !!! - - # # #
    return u, v, nConv
    

def angularDerivative(array0, array1, params):
    """
    This function computes the temporal partial derivative of array1 based on the method of finite differences.

    INPUT:
    :array0: (X x Y) dimensional array of one time-step before array1
    :array0: (X x Y) dimensional array of phase data whose temporal derivative we want to know

    OUTPUT:
    :Dt: (X x Y) dimensional array of temporal derivative
    """

    Dt = ( (array0 - array1 + np.ones(array1.shape)*np.pi)%2*np.pi ) - np.one(array1.shape)*np.pi

    return Dt

def spatialLaplacian(array, params):
    """
    This function computes the Laplacian of a matrix. 
    Method used is the 2-dimensional 5-point stencil method.
    NOTE: We exploit the torus - all-to-all connectivity of the model for calling the function-values

    INPUT: 
    :array: (X x Y) dimensional array of values
    :params: dictionary of parameters, only parameters that HAVE TO BE SET is dx & dy

    OUTPUT:
    :lap: (X x Y) dimensional array of spatial Laplacian
    """

    #for f-values
    f1 = np.roll(array, -1, axis=1)
    f2 = np.roll(array, 1, axis=1)
    f3 = np.roll(array, -1, axis=0)
    f4 = np.roll(array, 1, axis=0)
    f5 = -4*array

    lap = ( f1 + f2 + f3 + f4 + f5) * 1/(params.dx*params.dy)


    return lap
def spatialPartialDerivative(array, next_array, params):
    """
    This function computes the partical derivatives of a matrix into both directions. 
    Method used is the same as in Horn & Schink, 1981, "Determining Optical Flow", p. 189
    Difference: They assume the cube to parcellated into segments of unit length, our's have length dx*dy*dt
    NOTE: We exploit the torus - all-to-all connectivity of the model for calling the function-values

    INPUT: 
    :array: (X x Y) dimensional array of values
    :params: dictionary of parameters, only parameters that HAVE TO BE SET is dx, dy & dt

    OUTPUT:
    :Dx: (X x Y) dimensional array of spatial derivative of first order into x-direction
    :Dy: (X x Y) dimensional array of spatial derivative of first order into y-direction
    """

    # for x-direction
    fx1 = np.roll(array, 1, axis=1)
    fx2 = -array
    fx3 = np.roll(array, 1, axis=1)
    fx3 = np.roll(fx3, 1, axis=0)
    fx4 = -np.roll(array, 1, axis=0)
    fx5 = np.roll(next_array, 1, axis=1)
    fx6 = -next_array
    fx7 = np.roll(next_array, 1, axis=1)
    fx7 = np.roll(fx7, 1, axis=0)
    fx8 = -np.roll(next_array, 1, axis=0)


    # for y-direction
    fy1 = np.roll(array, 1, axis=0)
    fy2 = -array
    fy3 = np.roll(array, 1, axis=0)
    fy3 = np.roll(fy3, 1, axis=1)
    fy4 = -np.roll(array, 1, axis=1)
    fy5 = np.roll(next_array, 1, axis=0)
    fy6 = -next_array
    fy7 = np.roll(next_array, 1, axis=0)
    fy7 = np.roll(fy7, 1, axis=1)
    fy8 = -np.roll(next_array, 1, axis=1)

    Dx = ( fx1 + fx2 + fx3 + fx4 + fx5 + fx6 + fx7 + fx8) * ( 1/(4*params.dx*params.dy*params.dt) )
    Dy = ( fy1 + fy2 + fy3 + fy4 + fy5 + fy6 + fy7 + fy8) * ( 1/(4*params.dx*params.dy*params.dt) )

    return Dx, Dy

def temporalPartialDerivative(array0, array1, phase_data=False, params={}):
    """
    This function computes the temporal derivative at time step of array0, based on the method of finite differences

    INPUT:
    :array 0&1: (X x Y) dimensional arrays of 2 snapshots
    :phase_data: bool, True if data includes phase data (i.e. is angula), False if not -> need to normalize values by scaling by the overall mean
    :params: dictionary of parameters, only parameters that HAVE TO BE SET is dt
    """

    if phase_data:
        Dt = angularDerivative(array0, array1, params)
    else:
        ft1 = array1
        ft2 = -array0
        ft3 = np.roll(array1, 1, axis=0)
        ft4 = -np.roll(array0, 1, axis=0)
        ft5 = np.roll(array1, 1, axis=1)
        ft6 = -np.roll(array0, 1, axis=1)
        ft7 = np.roll(array1, 1, axis=0)
        ft7 = np.roll(ft7, 1, axis=1)
        ft8 = np.roll(array0, 1, axis=0)
        ft8 = np.roll(ft8, 1, axis=1)

        Dt = ( ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8 ) * (1 / (4*params.dx*params.dy*params.dt) )

    return Dt



def runOpticalFlow(data, alpha=0.1, beta=10, nSteps=100, 
                   phase_data=False, params={}):
    """
    This function runs the optical flow equations for every time step t=1,...T. 
    For the method, please see Townsend & Gong (2018) "Detection and analysis of spatiotemporal patterns in brain activity"
    The provided code in the publication is written in matlab, we translated it to python and adjusted it to our setting.

    INPUT:
    :data: (X x Y x T) is the 2-dimensional data over all time steps T: rows = X, columns = Y, time_steps = T
    :alpha: > 0 float, regularization parameter "weighs smoothenes constrained", reasonable range: [0.1, 20]
    :beta: >0 float, non-linear penalty constant "regulates the degree of non-linearity", beta>>1 is approxi a quadratic penalty
    :phase_data: bool, True if data includes phase data (i.e. is angula), False if not -> need to normalize values by scaling by the overall mean
    :params: dictionary of parameters of model, are used to access the grid (space) and compute the spatial derivatives

    OUTPUT:
    :vx: (X x Y x T) dimensional velocity vector field into x-direction
    :vy: (X x Y x T) dimensional velocity vector field into y-direction
    :nConv: integer, how many iterations in opticalFlowStep were necessary. If nConv >= nSteps it hasn't converged
    """

    #set parameters X, Y, T, initialise vx & vy
    xdim = data.shape[0]
    ydim = data.shape[1]
    tdim = data.shape[2]

    vx = np.zeros((xdim, ydim, tdim))
    vy = np.zeros((xdim, ydim, tdim))

    #normalise by scaling by the overall mean
    if not phase_data:
        data *= (1/np.mean(data))
        #and convert complex numbers to amplitudes (Frage an mich: warum sollte ich komplexe Zahlen haben, wenn Nicht phase_data?)
        if not np.all(np.isreal(data)):
            data = abs(data)

    #iterate over all time steps, except t=tdim, since we can not compute the partial derivatives there
    for t in range(0, tdim-1):

        #determine the spatial derives at time step t
        Dx, Dy = spatialPartialDerivative(data[:,:,t], data[:,:,int(t+1)], params)
        Dt = temporalPartialDerivative(data[:,:,t], data[:,:,t+1], phase_data, params)
      #  print('Dx: ', Dx)
      #  print('Dx.shape: ', Dx.shape)
      #  print('Dy: ', Dy)
      #  print('Dy.shape: ', Dy.shape)
      #  print('Dt: ', Dt)
      #  print('Dt.shape: ', Dt.shape)

        vx_t, vy_t, nConv = opticalFlowStep(data[:,:,t], alpha, beta, Dx, Dy, Dt, nSteps, params)

        vx[:,:,t] = vx_t
        vy[:,:,t] = vy_t

        if t%10==0:
            print('VVF at time step %i is calculated' %t)


    return vx, vy, nConv