import numpy as np
from scipy.sparse import diags, linalg
import logging

# # # # # - - - - - functions/tools - - - - - # # # # # 

def penalty(x, beta):
    return np.sqrt(x + beta**2)


def penaltyPrime(x, beta):
    return (2*np.sqrt(x + beta**2))**(-1)


def localAverage(array):
    """
    This function computes the local average of the array of input.
    Method used in Horn & Schink, 1981, "Determining Optical Flow", p. 190.
    """

    f1 = np.roll(array, -1, axis=0)
    f2 = np.roll(array, 1, axis=1)
    f3 = np.roll(array, 1, axis=0)
    f4 = np.roll(array, -1, axis=1)

    f5 = np.roll(array, -1, axis=0)
    f5 = np.roll(f5, -1, axis=1)
    f6 = np.roll(array, 1, axis=1)
    f6 = np.roll(f6, -1, axis=0)
    f7 = np.roll(array, 1, axis=0)
    f7 = np.roll(f7, 1, axis=1)
    f8 = np.roll(array, -1, axis=1)
    f8 = np.roll(f8, 1, axis=0)

    locAvg = (1/6) * ( f1 + f2 + f3 + f4 )  + (1/12) * ( f5 + f6 + f7 + f8 )

    return locAvg


def flowVelocity(array, localAvg, kappa=1):

    """
    This function computes the approximation of the Laplacian, which is called the 'flow velocity' in Horn & Schink, 1981, "Determining Optical Flow", p. 190.

    INPUT: 
    :array: (X x Y) dimensional array of values
    :localAverage: (X x Y) dimensional array of average values of :array:
    :kappa: scaling factor, default set to 1
    :params: dictionary of parameters, only parameters that HAVE TO BE SET is dx & dy

    OUTPUT:
    :flowVel: (X x Y) dimensional array of approximation of spatial Laplacian ('flow velocity')
    """

    flowVel = kappa * ( localAvg - array )

    return flowVel


# # # # # - - - - - derivatives - - - - - # # # # #

def angularDerivative(array0, array1):
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


def spatialLaplacian(array):
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

    lap = ( f1 + f2 + f3 + f4 + f5)

    return lap


def spatialPartialDerivative(array, next_array):
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

    Dx = ( fx1 + fx2 + fx3 + fx4 + fx5 + fx6 + fx7 + fx8) * (1/4) 
    Dy = ( fy1 + fy2 + fy3 + fy4 + fy5 + fy6 + fy7 + fy8) * (1/4) 

    return Dx, Dy


def temporalPartialDerivative(array0, array1, phase_data=False):
    """
    This function computes the temporal derivative at time step of array0, based on the method of finite differences

    INPUT:
    :array 0&1: (X x Y) dimensional arrays of 2 snapshots
    :phase_data: bool, True if data includes phase data (i.e. is angular), False if not -> need to normalize values by scaling by the overall mean
    :params: dictionary of parameters, only parameters that HAVE TO BE SET are dx, dy & dt
    """

    if phase_data:
        Dt = angularDerivative(array0, array1)
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

        Dt = ( ft1 + ft2 + ft3 + ft4 + ft5 + ft6 + ft7 + ft8 ) * (1/4)

    return Dt


# # # # # - - - - - Optical Flow - - - - - # # # # # 


def opticalFlowStep(array, alpha, beta, Dx, Dy, Dt, nSteps):
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

    #set annealing parameter, learning rate for fixed-point iteration
    epsilon = 1.1
    learnStep = 0.02
    epsilonMin = 0.1
    kappa = 1

    #initialise parameters
    u = np.zeros((array.shape))
    v = np.zeros((array.shape))
    ps = np.zeros((array.shape))
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
        ps_old = ps

        #compute new Ed, Es, Laplacians of u, v 
        Ed = Dx * u + Dy * v + Dt

        uAvg = localAverage(u)
        vAvg = localAverage(v)

        if nConv == 0:
            ux, uy, vx, vy = np.zeros((array.shape)), np.zeros((array.shape)), np.zeros((array.shape)), np.zeros((array.shape))
            psx, pxy = np.zeros((array.shape)), np.zeros((array.shape))
        else:
            ux, uy = spatialPartialDerivative(u_old, u)
            vx, vy = spatialPartialDerivative(v_old, v)
            psx, psy = spatialPartialDerivative(ps_old, ps)
        Es = ux**2 + uy**2 + vx**2 + vy**2 #def as in Horn and Schunk, 1981, p. 190

        pd = (1/alpha) * penaltyPrime(Ed**2, beta)
        ps = penaltyPrime(Es**2, beta)

        if nConv == 0:
            psx, psy = np.zeros((array.shape)), np.zeros((array.shape))
        else:
            psx, psy = spatialPartialDerivative(ps_old, ps)

        

        #compute the distance between the new Ed and old Ed (Es analogously) to check whether it has converged early
        diffEd = abs(Ed_old - Ed) / abs(Ed)
        diffEs = abs(Es_old - Es) / abs(Es) 
        if np.max(diffEs) < 0.1e-6 and np.max(diffEd) < 0.1e-6:
            break

        #reformulate equations s.t. we have a linear system of the Form Ax=b
        #doing this fully analogue to Townsend and Gong (2018) and Horn & Schink (1981).

        #print('All parameters inlcuding the convergence-check were done in opticalFlowStep.')

        # Set al necessary values for the iteration to simplify things
        psk = ps*kappa
       
        #set parameters X, Y, T, initialise vx & vy
        xdim = array.shape[0]
        ydim = array.shape[1]
        N = xdim * ydim

        psx, psy = spatialPartialDerivative(ps_old, ps)
#        print('psx & psx.shape: ', psx, psx.shape)

        #single parts of fixed-point iteration #TODO: check whether elementwise or matrix multiplication?!
        pdDx = pd * Dx
        pdDy = pd * Dy
        pdDxDy = pdDx * Dy
        pdDxDx = pdDx * Dx
        pdDyDy = pdDy * Dy


#        print('pdDx & pdDx.shape: ', pdDx, pdDx.shape)
#        print('pdDxDy & pdDxDy.shape: ', pdDxDy, pdDxDy.shape)

        gamma = pdDxDx + pdDyDy + psk
#        print('gamma & gamma.shape: ', gamma, gamma.shape)

        psgrad_ugrad = psx * ux + psy * uy
        psgrad_vgrad = psx * vx + psy * vy
#        print('psgrad_ugrad & psgrad_ugrad.shape: ', psgrad_ugrad, psgrad_ugrad.shape)

        Dxfraction = ( pdDxDx / psk )
        Dyfraction = ( pdDyDy / psk )


        #we now start the computation for the upper part of the fixed point iterations' fraction

        avgComp = (Dx * uAvg + Dy * vAvg + Dt)
#        print('avgComp & avgComp.shape: ', avgComp, avgComp.shape)

        bu1 = -pdDx * avgComp
        bu2 = Dyfraction * psgrad_ugrad
        bu3 = psgrad_ugrad
        bu4 = - pdDxDy * psgrad_vgrad

        bv1 = -pdDy * avgComp
        bv2 = Dxfraction * psgrad_vgrad
        bv3 = psgrad_vgrad
        bv4 = pdDxDy * psgrad_ugrad

        gamma_inverse = 1 / gamma
#        print('gamma_inverse & gamma_inverse.shape: ', gamma_inverse, gamma_inverse.shape)

        u = (1-epsilon) * uAvg + epsilon * (bu1 + bu2 + bu3 + bu4) * gamma_inverse
        v = (1-epsilon) * vAvg + epsilon * (bv1 + bv2 + bv3 + bv4) * gamma_inverse

        if epsilon > epsilonMin:
            epsilon -=learnStep

        nConv +=1

#        if nConv%10==0:
#            print('inside step: VVF at time step %i is calculated' %nConv)

    return u, v, nConv


def runOpticalFlow(data, alpha=0.1, beta=10, nSteps=100, 
                   phase_data=False):
    """
    This function runs the optical flow equations for every time step t=1,...T. 
    For the method, please see Townsend & Gong (2018) "Detection and analysis of spatiotemporal patterns in brain activity"
    The provided code in the publication is written in matlab, we used it as a basis for most of this implementation and translation to python.
    It is adjusted and partly changed. Reasons are both personal coding/implementation preferences and necessary changes due to a language-matlab-python barrier.
    Further, we implemented a structure such that anyone can easily take this code and adjust it to their own setting 
    (e.g. if one wants to use a differen penalty function, or a different approximation of partial derivatives, or including boundary conditions, ...)

    INPUT:
    :data: (X x Y x T) is the 2-dimensional data over all time steps T: rows = X, columns = Y, time_steps = T
    :alpha: > 0 float, regularization parameter "weighs smoothenes constrained", reasonable range: [0.1, 20]
    :beta: >0 float, non-linear penalty constant "regulates the degree of non-linearity", beta>>1 is approxi a quadratic penalty
    :phase_data: bool, True if data includes phase data (i.e. is angula), False if not -> need to normalize values by scaling by the overall mean

    OUTPUT:
    :vx: (X x Y x T) dimensional velocity vector field into x-direction
    :vy: (X x Y x T) dimensional velocity vector field into y-direction
    :nConv: integer, how many iterations in opticalFlowStep were necessary. If nConv >= nSteps it hasn't converged
    """

    #set parameters X, Y, T, initialise vx & vy
    xdim = data.shape[0]
    ydim = data.shape[1]
    tdim = data.shape[2] -1 #omit last time step for partial detivative approximations

    vx = np.zeros((xdim, ydim, tdim))
    vy = np.zeros((xdim, ydim, tdim))

    #normalise by scaling by the overall mean
    if not phase_data:
        data *= (1/np.mean(data))
        #and convert complex numbers to amplitudes (Frage an mich: warum sollte ich komplexe Zahlen haben, wenn Nicht phase_data?)
        if not np.all(np.isreal(data)):
            data = abs(data)

    #iterate over all time steps, except t=tdim, since we can not compute the partial derivatives there
    for t in range(0, tdim):

        #determine the spatial derives at time step t
        Dx, Dy = spatialPartialDerivative(data[:,:,t], data[:,:,int(t+1)])
        Dt = temporalPartialDerivative(data[:,:,t], data[:,:,t+1], phase_data)

        vx_t, vy_t, nConv = opticalFlowStep(data[:,:,t], alpha, beta, Dx, Dy, Dt, nSteps)

        vx[:,:,t] = vx_t
        vy[:,:,t] = vy_t

        if t%10==0:
            print('VVF at time step %i is calculated' %t)


    return vx, vy, nConv