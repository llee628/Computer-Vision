#!/usr/bin/python

import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pdb


def colormapArray(X, colors):
    """
    Basically plt.imsave but return a matrix instead

    Given:
        a HxW matrix X
        a Nx3 color map of colors in [0,1] [R,G,B]
    Outputs:
        a HxW uint8 image using the given colormap
    """

    ## if all values in X is zero, return directly
    if np.all((X == 0)):
        return X.astype(np.uint8)

    #Uncomment for dataset2
    #X = X**0.5

    H = X.shape[0]
    W = X.shape[1]
    N = colors.shape[0]

    output = np.zeros((H,W,3))
    raw_idx = X.copy()

    #deal with nan edge case
    if True in np.isnan(raw_idx):
        print("X contains nan!!")
        nan_idx = np.argwhere(np.isnan(raw_idx))
        for nan_i in nan_idx:
            raw_idx[nan_i[0], nan_i[1]] = 0
        
    #breakpoint()

    vmax = np.max(raw_idx)
    vmin = np.min(raw_idx)
    raw_idx = raw_idx - vmin
    #breakpoint()
    raw_idx = raw_idx*(N-1)/(vmax-vmin)
    raw_idx = raw_idx.astype(int)
    #breakpoint()
    for i in range(H):
        for j in range(W):
            raw_color = raw_idx[i,j]
            output[i,j] = colors[raw_color,:]

    #handle float number to uint8
    # float num: 0~1; uint8: 0~255
    output = output*255

    return output.astype(np.uint8)


if __name__ == "__main__":
    colors = np.load("mysterydata/colors.npy")

    '''
    
    Uncomment each data one by one to test different dataset     
    
    '''
    #data = np.load("mysterydata/mysterydata.npy")
    data = np.load("mysterydata/mysterydata2.npy")
    #data = np.load("mysterydata/mysterydata3.npy")
    #data = np.load("mysterydata/mysterydata4.npy")
    for i in range(data.shape[2]):
        false_color_img = colormapArray(data[:,:,i], colors)
        plt.imshow(false_color_img)
        plt.show()
    
    


    pdb.set_trace()
