import os
import pdb
#import cv2
import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use same padding (mode = 'reflect'). Refer docs for further info.

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)

def gaussian_kernel_generator(sigma, size=(3,3)):
    output = np.zeros(size)
    H = size[0]
    W = size[1]
    #breakpoint()
    m = H//2
    n = W//2

    for i in range(-m, m+1):
        for j in range(-n, n+1):
            output[i+m,j+n] = (1.0/(2*(np.pi)*(sigma**2)))*np.exp(-((i**2) + (j**2))/(2*(sigma**2)))

    #breakpoint()
    #smoothing filter
    output_sum = np.sum(output)
    output = output/output_sum
    #breakpoint()
    return output


def gaussian_filter(image, sigma):
    """
    Given an image, apply a Gaussian filter with the input kernel size
    and standard deviation

    Input
      image: image of size HxW
      sigma: scalar standard deviation of Gaussian Kernel

    Output
      Gaussian filtered image of size HxW
    """
    H, W = image.shape
    # -- good heuristic way of setting kernel size
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    # Ensure that the kernel size isn't too big and is odd
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1
    # TODO implement gaussian filtering of size kernel_size x kernel_size
    # Similar to Corner detection, use scipy's convolution function.
    # Again, be consistent with the settings (mode = 'reflect').
    kernel = gaussian_kernel_generator(sigma, (kernel_size,kernel_size))
    output = scipy.ndimage.convolve(image, kernel, mode='reflect')
    return output

def resize_image(image, scale_percent=200):
    width = int(image.shape[1]*scale_percent/100)
    height = int(image.shape[0]*scale_percent/100)
    dim = (width, height)

    resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized





def main():
    image = read_img('polka.png')
    # import pdb; pdb.set_trace()
    # Create directory for polka_detections
    if not os.path.exists("./polka_detections"):
        os.makedirs("./polka_detections")

    # -- TODO Task 7: Single-scale Blob Detection --
    
    # (a), (b): Detecting Polka Dots
    # First, complete gaussian_filter()
    print("Detecting small polka dots")
    # -- Detect Small Circles
    #sigma_1, sigma_2 = None, None
    sigma_1 = 4.5/np.sqrt(2)
    sigma_2 = 1.5*sigma_1
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_small = gauss_2 - gauss_1  # to implement

    # visualize maxima
    maxima = find_maxima(DoG_small, k_xy=10)
    print("There are",len(maxima), "maxima being detected")
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')

    #breakpoint()
    # -- Detect Large Circles
    print("Detecting large polka dots")
    sigma_1 = 11/np.sqrt(2)
    sigma_2 = 1.6*sigma_1
    gauss_1 = gaussian_filter(image, sigma_1)  # to implement
    gauss_2 = gaussian_filter(image, sigma_2)  # to implement

    # calculate difference of gaussians
    DoG_large = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_large, k_xy=10)
    print("There are",len(maxima), "maxima being detected")
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_large.png')

    #breakpoint()
    
    # # -- TODO Task 8: Cell Counting --
    print("Detecting cells")

    # Detect the cells in any four (or more) images from vgg_cells
    image1 = read_img('./cells/043cell.png')
    image2 = read_img('./cells/057cell.png')
    image3 = read_img('./cells/061cell.png')
    image4 = read_img('./cells/166cell.png')

    # Create directory for cell_detections
    if not os.path.exists("./cell_detections"):
        os.makedirs("./cell_detections")

    #image1
    
    sigma_1 = 5.0/np.sqrt(2)
    sigma_2 = 7.0*sigma_1
    gauss_1 = gaussian_filter(image1, sigma_1)
    gauss_2 = gaussian_filter(image1, sigma_2)

    # calculate difference of gaussians
    DoG_043 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_043, k_xy=7)
    print("There are",len(maxima), "cells being detected")
    visualize_scale_space(DoG_043, sigma_1, sigma_2 / sigma_1,
                          './cell_detections/cell_detections_DoG_043.png')
    visualize_maxima(image1, maxima, sigma_1, sigma_2 / sigma_1,
                     './cell_detections/cell_detections_043.png')
    
    
    #image2
    
    sigma_1 = 3.0/np.sqrt(2)
    sigma_2 = 8.3*sigma_1
    gauss_1 = gaussian_filter(image2, sigma_1)
    gauss_2 = gaussian_filter(image2, sigma_2)

    # calculate difference of gaussians
    DoG_057 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_057, k_xy=7)
    print("There are",len(maxima), "cells being detected")
    visualize_scale_space(DoG_057, sigma_1, sigma_2 / sigma_1,
                          './cell_detections/cell_detections_DoG_057.png')
    visualize_maxima(image2, maxima, sigma_1, sigma_2 / sigma_1,
                     './cell_detections/cell_detections_057.png')
    
    
    #image3
    
    #image3 = resize_image(image3, 125)
    sigma_1 = 5.0/np.sqrt(2)
    sigma_2 = 6.9*sigma_1
    gauss_1 = gaussian_filter(image3, sigma_1)
    gauss_2 = gaussian_filter(image3, sigma_2)

    # calculate difference of gaussians
    DoG_061 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_061, k_xy=11)
    print("There are",len(maxima), "cells being detected")
    visualize_scale_space(DoG_061, sigma_1, sigma_2 / sigma_1,
                          './cell_detections/cell_detections_DoG_061.png')
    visualize_maxima(image3, maxima, sigma_1, sigma_2 / sigma_1,
                     './cell_detections/cell_detections_061.png')
    
    

    #image4
    
    sigma_1 = 5.0/np.sqrt(2)
    sigma_2 = 5.9*sigma_1
    gauss_1 = gaussian_filter(image4, sigma_1)
    gauss_2 = gaussian_filter(image4, sigma_2)

    # calculate difference of gaussians
    DoG_166 = gauss_2 - gauss_1  # to implement

    # visualize maxima
    # Value of k_xy is a sugguestion; feel free to change it as you wish.
    maxima = find_maxima(DoG_166, k_xy=6)
    print("There are",len(maxima), "cells being detected")
    visualize_scale_space(DoG_166, sigma_1, sigma_2 / sigma_1,
                          './cell_detections/cell_detections_DoG_166.png')
    visualize_maxima(image4, maxima, sigma_1, sigma_2 / sigma_1,
                     './cell_detections/cell_detections_166.png')
    
    





if __name__ == '__main__':
    main()
