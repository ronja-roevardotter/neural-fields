import numpy as np
import matplotlib.pyplot as plt


def interpolateAlongLine(start_point, end_point, num_points=100):
    """
    Interpolate points along a line segment defined by two points.
    
    Args:
    :start_point: (x, y) coordinates of the starting point, tuple
    :end_point: (x, y) coordinates of the ending point, tuple
    :num_points: Number of points to interpolate along the line segment, integer
    
    Returns:
    :interpolatex_x, interpolated_y: stack of interpolated points as a 2D array (each row represents a point with x, y coordinates).

    Disclaimer: this function was written with the help of chatGPT, but thoroughly tested.
    """
    x_start, y_start = start_point
    x_end, y_end = end_point
    
    t = np.linspace(0, 1, num_points)
    interpolated_x = np.interp(t, [0, 1], [x_start, x_end])
    interpolated_y = np.interp(t, [0, 1], [y_start, y_end])
    
    return np.column_stack((interpolated_x, interpolated_y))

def interpolatePath(path, num_points=100):
    """
    Interpolate points along a path.
    
    INPUT:
    :contour_paths: List of Path objects representing the contour lines (pyplot path object, list)
    :num_points: interpolation-rate, integer
    
    OUTPUT:
    :ninterpolated)point: interpolated points as a 2D array (each row represents a point with x, y coordinates).

    Disclaimer: this function was written with the help of chatGPT, but thoroughly tested.
    """
    interpolated_points = []
    
    vertices = path.vertices
    codes = path.codes
    num_segments = len(codes)
    
    for i in range(num_segments-1):
        start_point = vertices[i]
        end_point = vertices[i + 1]
        points_on_line = interpolateAlongLine(start_point, end_point, num_points)
        interpolated_points.extend(points_on_line)
    
    return np.array(interpolated_points)


def findContourPaths(array):
    """This function simply returns the contour paths (interpolated) over an nxm-dimensional array

    INPUT: 
    :array: nxm-dimensional arrat of scalar values

    OUTPUT:
    :contours: every contour line found in array, (pyplot path object, list)
    """

    #Define the levels for contour-lines
    levels=[0]

    #exploitation of plotting function that inherits the computation of contour lines
    plot = plt.contour(array, levels)
    #extraction of contour paths
    contours = plot.collections[0].get_paths()

    return contours


def contourMatrix(array, contour_paths):
    """
    Create a binary matrix indicating the positions of contour lines in the original nxm-dimensional array.
    
    INPUT:
    :array: nxm-dimensional array for which the contours were determined
    :contour_paths: List of Path objects representing the contour lines (pyplot path object, list)
    
    OUTPUT:
    :contour_matrix: binary matrix where 1 indicates the positions of contour lines and 0 otherwise.

    Disclaimer: this function was written with the help of chatGPT, but thoroughly tested.
    """
    #initialize binary matrix with zeros
    contour_matrix = np.zeros_like(array, dtype=int)
    
    #iterate over each contour path
    for path in contour_paths:
        # Interpolate points along the contour path
        interpolated_points = interpolatePath(path)
        
        # Round interpolated points to nearest integer coordinates
        interpolated_points = np.round(interpolated_points).astype(int)
        
        # Clip coordinates to stay within array bounds
        interpolated_points[:, 0] = np.clip(interpolated_points[:, 0], 0, array.shape[0] - 1)
        interpolated_points[:, 1] = np.clip(interpolated_points[:, 1], 0, array.shape[1] - 1)
        
        # Set positions of contour lines to 1 in the binary matrix
        contour_matrix[interpolated_points[:, 0], interpolated_points[:, 1]] = 1
    
    return contour_matrix


def criticalPoints(contourMatrix):
    """
    Turns the nxm-dimensional matrix of contour-line positions to aa 2x(#critical_points) array.

    INPUT:
    :contourMatrix: nxm-dimensional array with value 1 if contour line passes through coordinate location, 0 otherwise

    OUTPUT: 
    :criticalPoints: 2 x (#critical points) array with the first column contatining the x-, the second the y-component of the critical point.

    """

    criticalPoints = np.argwhere(contourMatrix==1)

    return criticalPoints


def classifyCriticalPoints(rowcoords, colcoords, u, v):
    """
    Find and classify the critical points in the vector field defined by u and v.

    INPUT:
    :u: X-component of the vector field, array
    :v: Y-component of the vector field, array

    OUTPUT:
    :rowcoords: Row coordinates of each critical point, array
    :colcoords: Column coordinates of each critical point, array
    :patType: Type of critical point present, list
    :jacobians: Estimated Jacobian at each critical point, array

    Disclaimer: This beautiful function is a translation from Townsend & Gong (2018) and their provided optical-flow pipeline: 
    https://github.com/BrainDynamicsUSYD/NeuroPattToolbox/blob/master/PatternDetection/classifyCrit.m

    """
    #count amount of critical points
    num_points = len(rowcoords)
    
    jacobians = np.zeros((2, 2, num_points))
    patType = []

    if num_points > 0:
        vxx, vxy = np.gradient(u)
        vyx, vyy = np.gradient(v)

    for ic in range(num_points):
        ix, iy = rowcoords[ic], colcoords[ic]
        corners = np.array([[np.floor(ix), np.ceil(ix)], [np.ceil(iy), np.floor(iy)]])
        corners = corners.astype(int)

        #estimation of Jacobian at the critical point through bilinear interpolation
        ixdec = ix - np.floor(ix)
        iydec = iy - np.floor(iy)
        dxx = np.dot([1 - ixdec, ixdec], np.dot(vxx[corners[0], corners[1]], [1 - iydec, iydec]))
        dxy = np.dot([1 - ixdec, ixdec], np.dot(vxy[corners[0], corners[1]], [1 - iydec, iydec]))
        dyx = np.dot([1 - ixdec, ixdec], np.dot(vyx[corners[0], corners[1]], [1 - iydec, iydec]))
        dyy = np.dot([1 - ixdec, ixdec], np.dot(vyy[corners[0], corners[1]], [1 - iydec, iydec]))

        ijac = np.array([[dxx[0], dxy[0]], [dyx[0], dyy[0]]])
        jacobians[:, :, ic] = ijac

        #classify critical point by its Jacobian
        jac_det = np.linalg.det(ijac)
        jac_trace = np.trace(ijac)
        
        if jac_det < 0:
            itype = 'saddle'
        elif jac_trace ** 2 > 4 * jac_det:
            if jac_trace < 0:
                itype = 'stableNode'
            else:
                itype = 'unstableNode'
        else:
            if jac_trace < 0:
                print('WE FOUND A STABLE FOCUS')
                itype = 'stableFocus'
            else:
                itype = 'unstableFocus'
        patType.append(itype)

    return rowcoords, colcoords, patType, jacobians

def avgNormVelocity(u, v):
    """"
    Determines the average normaliced velocity for the velocity vector field w(x,y,t) = (u,v). 
    The statistic is equivalent to the phase gradient directionality (see Townsend & Gong (2018)) with its' value reaching 1 as velocity vectors align (i.e. synchronize in direction and velocity)

    INPUT:
    :u, v: nxm-dimensional arrays of velocity vector fields of the x- and y-component respectively, both at time step t

    OUTPUT:
    :phi_t: the averaged normalized velocity at time step t, nxm-dimensional array
    """
    
    #first, compute the velocities over the field based on the x- and y-components u & v
    w = np.sqrt( u ** 2 + v ** 2)

    nominator = np.sqrt(np.sum(w) ** 2)
    denominator = np.sum(np.sqrt(w ** 2))

    phi_t = nominator/denominator
    
    return phi_t

def sequentialAvgNormVelocity(us, vs):
    """"
    Determine the average order paramter phi over a time series of velocity vector fields in x- (i.e. us) and y- (i.e. vs) component.

    INPUT:
    :us, vs: time series of x- and y-component vvfs (T x X x Y)-dimensional, array of arrays

    OUTPUT:
    :phi: average over time of average normalized veloity per time frame 
    """
    phis = np.zeros(len(us))
    count = 0

    for u, v in zip(us, vs):
        phi_t = avgNormVelocity(u, v)
        phis[count] = phi_t
        count+=1

    phi = np.mean(phis)

    return phi


# # # # # - - - - - Properties - - - - - # # # # #

