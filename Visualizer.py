###################################################################################
''' imports '''
import numpy as np
import matplotlib.pyplot as plt

###################################################################################

def plt_show(pattern):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(pattern > 0.5, edgecolor='k')
    plt.show()