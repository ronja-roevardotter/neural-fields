import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import matplotlib.ticker as ticker
from matplotlib import cm


def plot2DiscreteMaps(df, xaxis='I_e', yaxis='I_i'):
    
    p_colors = cm.get_cmap('Accent', 4)
    
    stabis = df.pivot_table('stability', columns=xaxis, index=yaxis)
    turings = df.pivot_table('turing', columns=xaxis, index=yaxis)
    p_randoms = df.pivot_table('p_random', columns=xaxis, index=yaxis)
        
    p_turings = df.pivot_table('p_turing', columns=xaxis, index=yaxis)
    
    fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6),dpi=120)

    ax1.imshow(p_randoms, origin='lower', vmin=1, vmax=4, aspect='auto', cmap=p_colors)
    ax1.contour(stabis, origin='lower', vmin=0, vmax=2, levels=1, cmap='YlGnBu', linewidths=3)
    ax1.contour(turings, origin='lower', vmin=0, vmax=1, levels=0, colors='black', linewidths=3, linestyles='dashed')
    ax1.set(title='Random initialization')

    ax2.imshow(p_turings, origin='lower', vmin=1, vmax=4, aspect='auto', cmap=p_colors)
    ax2.contour(stabis, origin='lower', vmin=0, vmax=2, levels=1, cmap='YlGnBu', linewidths=3)
    ax2.contour(turings, origin='lower', vmin=0, vmax=1, levels=0, colors='black', linewidths=3, linestyles='dashed')
    ax2.set(title='Initialization in Turing unstable FP')
    
    
    pos = ax2.imshow(p_turings, vmin=1, vmax=4, origin='lower', aspect='auto', cmap=p_colors)
    
    # Add minorticks on the colorbar to make it easy to read the
    # values off the colorbar.
    #cbar = fig.colorbar(pos, ax=ax2, extend='both')
    
    nmb_labels = 5
                
    idx_x = np.linspace(0,len(stabis.columns.values)-1, nmb_labels).astype(int) # np.round(np.linspace(0, len(a[0]) - 1, nmb_labels)).astype(int)
    idx_y = np.linspace(0,len(stabis.index.values)-1, nmb_labels).astype(int) # np.round(np.linspace(0, len(a) - 1, nmb_labels)).astype(int)
        
    xliste= np.round(stabis.columns.values, decimals=2)[idx_x]
    yliste= np.round(stabis.index.values, decimals=2)[idx_y]
    
    #xliste = np.linspace(xaxis[0],xaxis[-1],nmb_labels)
    #yliste = np.linspace(yaxis[-1],yaxis[0],nmb_labels)
    
    xlabels=list('%.1f'%(e) for e in xliste)
    ylabels=list('%.1f'%(e) for e in yliste)
    
    for ax in [ax1, ax2]:
        ax.xaxis.set_major_locator(ticker.LinearLocator(nmb_labels))
        ax.set_xticklabels(labels=xlabels, fontsize=20)
        ax.yaxis.set_major_locator(ticker.LinearLocator(nmb_labels))
        ax.set_yticklabels(labels=ylabels, fontsize=20)
        
        ax.set_xlabel(r'$%s$' %xaxis, fontsize=20)
        ax.set_ylabel(r'$%s$' %yaxis, fontsize=20, labelpad=10, rotation=0)
        
        ax.label_outer()
    
    mini = 1 #math.floor(vmin*10)/10 
    maxi = 4
    cbar_ticks=np.linspace(mini,maxi,4)
    cbar_ticks=np.around(cbar_ticks, decimals=0)
    cbar_labels=['stat', 'temp', 'spat', 'spatiotemp']
    
    # Create colorbar
    cbar = ax2.figure.colorbar(pos, ax=ax2, ticks=cbar_ticks)
    cbar.ax.set_ylabel('pattern-type', rotation=-90, va="bottom")
    cbar.ax.set_yticklabels(cbar_labels)
        
        
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    
    cbar.minorticks_on()
    
    
    plt.show()
    
    
def plotQuadrants(xvalues, yvalues, axis_max = 0.4, colors=['darkgreen', 'green', 'forestgreen', 'darkseagreen', 'limegreen', 'yellowgreen', 'greenyellow']):
    
    
    rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
          "xtick.major.size" : 5, "ytick.major.size" : 5,}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(10,10))
        
        count=0
        for vals in np.stack((xvalues, yvalues), axis=1):
            
            ax.plot(vals[0], vals[1], color=colors[count], lw=2)
            count+=1
        
        ax.scatter([0, 0, axis_max, -axis_max], [axis_max, -axis_max, 0, 0], s=0.1)
        
        
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        
        
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        
        # make arrows
        ax.plot((1), (0), ls="", marker=">", ms=10, color="k",
                transform=ax.get_yaxis_transform(), clip_on=False)
        ax.plot((0), (1), ls="", marker="^", ms=10, color="k", 
                transform=ax.get_xaxis_transform(), clip_on=False)
        
        plt.plot
        

def plotTraceDeterminant(xvalues, yvalues, k, 
                         colors=['darkgreen', 'green', 'forestgreen', 'darkseagreen', 'limegreen', 'yellowgreen', 'greenyellow'], 
                         ls=["", '--', '-.', ':', (0, (3, 10, 1, 10, 1, 10)), (0, (3, 1, 1, 1, 1, 1))]):
    
    plt.figure(figsize=(10,10))
    
    zero = np.zeros(len(k))
    count=0
    for vals in np.stack((xvalues, yvalues), axis=1):
            
            plt.plot(k, vals[0], k, vals[1], color=colors[count], lw=2)#, ls=ls[count])
            count+=1
            
    plt.plot(k, zero, c='black')