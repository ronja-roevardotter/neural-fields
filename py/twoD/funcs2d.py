import numpy as np
from matplotlib.animation import FuncAnimation


def mkAnimation(ue, file_name='activity.mp4', interval=50, fps=10):
    

    # Create a figure and axis
    fig, ax = plt.subplots()
    
    # Define the update function
    def update(frame):
        ax.imshow(ue[frame].T, cmap='viridis', vmin=0, vmax=1)  # Update the plot for each frame
        return ax
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=len(ue), interval=interval)
    
    # Save the animation as a video file
    return anim #anim.save(file_name, fps=fps)  # Adjust fps as needed
    
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # #
# # # - - - Below are functions that can be used to gather information about the kernels w.r.t. the defined space - - - # #
# # # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - # #

def circleLimit_Gaussian(params):
    """
    This function computes the radiuses for the Gaussian kernels w_e(x) and w_i(x), where the positive feedback becomes negative 
    (i.e. at which radius the kernel crosses the zero-line, w(||r||)=0). The function is deterministic.

    INPUT:
    :params: dictionary of model parameters, since weights and kernel-widths are necessary

    OUTPUT:
    :re: radius for w_e, float
    :ri: radius for w_i, float
    """

    re = (-2*(params.sigma_i**2)*
          np.log((params.w_ee*params.sigma_i**2)
                 /(params.w_ei*params.sigma_e**2))
          /(1-(params.sigma_i**2/params.sigma_e**2)))
    ri = (-2*(params.sigma_i**2)*
          np.log((params.w_ie*params.sigma_i**2)
                 /(params.w_ii*params.sigma_e**2))
          /(1-(params.sigma_i**2/params.sigma_e**2)))
    
    return re, ri

def withinCircle(params, re, ri):
      """"
        This function computes the total negative (g_minus) and total positive (g_plus) input per population. 
        Note: It does so by taking the pre-computed kernel-arrays, not the kernel functions in kernels2d.py, since we want 
              to know the total input for this setting (i.e. also considering the chosen discretization), and not in theory.
        Note: Doesn't work for rectangular kernel anymore. Only for Gaussian and Exponential.
        
        INPUT:
        :params: dictionary of model parameters, since weights and kernel-widths are necessary
        :re: radius for positive feedback in excitatory population, float
        :ri: radius for positive feedback in inhibitory population, float

        OUTPUT:
        :gE_plus: total positive input to exc, float
        :gE_minus: total negative input to exc, float
        :gI_plus: total positive input to inh, float
        :gI_minus: total negative input to inh, float

      """
      gE_plus = 0
      gE_minus = 0
      gI_plus = 0
      gI_minus = 0

      for i, row in enumerate(params.distx):
            for j, entry in enumerate(row):
                  if entry <= re:
                      gE_plus += params.w_ee*params.ke[i,j]-params.w_ei*params.ki[i,j]
                  else: 
                      gE_minus += params.w_ee*params.ke[i,j]-params.w_ei*params.ki[i,j]

                  if entry <= ri:
                      gI_plus += params.w_ie*params.ke[i,j]-params.w_ii*params.ki[i,j]
                  else: 
                      gI_minus += params.w_ie*params.ke[i,j]-params.w_ii*params.ki[i,j]
      return gE_plus, gE_minus, gI_plus, gI_minus


def covered_space(space, dx, dy, r, kernel, thresh):
    """"
    This function determines, how much of the simulated space is covered by positive vs. negative feedback. 
    Negative feedback up to kernel values of 0.1e-15 are considered.

    INPUT:
    :space: nxm-dimensional array of distance values in the space to the point of origin
    :dx, dy: integration space constants
    :r: radius that indicates where the positive turns into negative input
    :kernel: nxm-dimensional array of kernel values to identify, where values of abs(negative feedback) < thresh

    OUTPUT:
    :pos_coverage: amount of space covered by positive kernel-feedback, float
    :neg_coverage: amount of space covered by abs(negativ kernel-feedback) < thresh, float
    :coverage_matrix: matrix with 1, if positivly covered, -1, if negatively covered, 0 otherwise, nxm-dimensional array
    """

    pos_coverage = 0
    neg_coverage = 0
    coverage_matrix = np.zeros(space.shape)


    for i, row in enumerate(space):
        for j, entry in enumerate(row):
            if entry <= r:
                pos_coverage += abs(space[i][j])*dx*dy
                coverage_matrix[i][j] = 1
            elif entry > r and abs(kernel[i][j]) >= thresh:
                neg_coverage += abs(space[i][j])*dx*dy
                coverage_matrix[i][j] = -1
    
    return pos_coverage, neg_coverage, coverage_matrix
