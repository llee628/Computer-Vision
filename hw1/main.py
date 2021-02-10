"""
Starter code for EECS 442 W21 HW1
"""
import os
import cv2
import numpy as np
import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt
import pdb
from util import generate_gif, renderCube



def rotX(theta):
    """
    Generate 3D rotation matrix about X-axis
    Input:  theta: rotation angle about X-axis
    Output: Rotation matrix (3 x 3 array)
    """
    Rx_theta = np.zeros((3,3))

    Rx_theta[0,0] = 1
    Rx_theta[1,1] = np.cos(theta)
    Rx_theta[1,2] = -(np.sin(theta))
    Rx_theta[2,1] = np.sin(theta)
    Rx_theta[2,2] = np.cos(theta)

    return Rx_theta
    


def rotY(theta):
    """
    Generate 3D rotation matrix about Y-axis
    Input:  theta: rotation angle along y-axis
    Output: Rotation matrix (3 x 3 array)
    """
    Ry_theta = np.zeros((3,3))

    Ry_theta[0,0] = np.cos(theta)
    Ry_theta[0,2] = np.sin(theta)
    Ry_theta[1,1] = 1
    Ry_theta[2,0] = -(np.sin(theta))
    Ry_theta[2,2] = np.cos(theta)
    #breakpoint()

    return Ry_theta

def degreeToRadian(theta):
    """
    Transfer degree to radian
    """
    return theta*np.pi/180

def task1_1():
    Ry_theta_list = []

    for degree in range(361):
        Ry_theta_list.append(rotY(degreeToRadian(degree)))

    generate_gif(Ry_theta_list)

def task1_2():
    theta = np.pi/4
    Rx_theta = rotX(theta)
    Ry_theta = rotY(theta)
    
    renderCube(R = Ry_theta.dot(Rx_theta), file_name='part1_2_1.png')
    renderCube(R = Rx_theta.dot(Ry_theta), file_name='part1_2_2.png')

def task1_3():
    theta1 = degreeToRadian(35.5)
    theta2 = degreeToRadian(45)
    Rx_theta = rotX(theta1)
    Ry_theta = rotY(theta2)
    
    renderCube(R=Rx_theta.dot(Ry_theta), file_name='part1_3.png')


def part1():
    # TODO: Solution for Q1
    # Task 1: Use rotY() to generate cube.gif
    
    task1_1()



    # Task 2:  Use rotX() and rotY() sequentially to check
    # the commutative property of Rotation Matrices
    
    task1_2()


    
    # Task 3: Combine rotX() and rotY() to render a cube 
    # projection such that end points of diagonal overlap
    # Hint: Try rendering the cube with multiple configrations
    # to narrow down the search region
    task1_3()

    pass


def split_triptych(trip):
    """
    Split a triptych into thirds
    Input:  trip: a triptych (H x W matrix)
    Output: R, G, B martices
    """
    H = trip.shape[0]
    W = trip.shape[1]
    #breakpoint()

    
    # TODO: Split a triptych into thirds and
    if H%3 == 0:
        R, G, B = trip[int(2*H/3):H,:], trip[int(H/3):int(2*H/3),:], trip[0:int(H/3),:]
    elif H%3 == 1:
        R, G, B = trip[int(2*H/3):H-1,:], trip[int(H/3):int(2*H/3),:], trip[0:int(H/3),:]
    else:
        R, G, B = trip[int(2*H/3):H-2,:], trip[int(H/3):int(2*H/3),:], trip[0:int(H/3),:] 
    #breakpoint()
    # return three channels as numpy arrays
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    a = ch1.reshape(ch1.size,1)
    b = ch2.reshape(ch2.size,1)
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)

    output = (a.T) @ b
    #breakpoint()

    return output[0,0]

def cross_correlation(ch1, ch2):
    """
    Without normalize
    """
    a = ch1.reshape(ch1.size,1)
    b = ch2.reshape(ch2.size,1)
    output = (a.T) @ b
    return output[0,0]


def best_offset(ch1, ch2, metric, Xrange=np.arange(-10, 10), 
                Yrange=np.arange(-10, 10)):
    """
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
            metric: similarity measure between two channels
            Xrange: range to search for optimal offset in vertical direction
            Yrange: range to search for optimal offset in horizontal direction
    Output: optimal offset for X axis and optimal offset for Y axis

    Note: Searching in Xrange would mean moving in the vertical 
    axis of the image/matrix, Yrange is the horizontal axis 
    """
    # TODO: Use metric to align ch2 to ch1 and return optimal offsets
    best_similar_X = 0
    best_similar_Y = 0

    for i in Xrange:
        temp = np.roll(ch2,i, axis=0)
        H = temp.shape[0]
        #breakpoint()

        if i < 0:
            similar_x = metric(ch1[0:H+i,:],temp[0:H+i,:])
        else:
            similar_x = metric(ch1[i:,:], temp[i:,:])
        
        if similar_x > best_similar_X:
            best_similar_X = similar_x
            best_offset_X = i
        #breakpoint()

    #breakpoint()
    for i in Yrange:
        temp = np.roll(ch2, i, axis=1)
        W = temp.shape[1]
        #breakpoint()

        if i < 0:
            similar_y = metric(ch1[:,0:W+i], temp[:,0:W+i])
        else:
            similar_y = metric(ch1[:,i:], temp[:,i:])

        if similar_y > best_similar_Y:
            best_similar_Y = similar_y
            best_offset_Y = i

    #breakpoint()
    return best_offset_X, best_offset_Y



    


def align_and_combine(R, G, B, metric):
    """
    Input:  R: red channel
            G: green channel
            B: blue channel
            metric: similarity measure between two channels
    Output: aligned RGB image 
    """
    # TODO: Use metric to align the three channels 
    # Hint: Use one channel as the anchor to align other two

    #fix R
    G_offset_X, G_offset_Y = best_offset(R, G, metric, np.arange(-15,15), np.arange(-15,15))
    B_offset_X, B_offset_Y = best_offset(R, B, metric, np.arange(-15,15), np.arange(-15,15))
    G = np.roll(G, G_offset_X, axis=0)
    G = np.roll(G, G_offset_Y, axis=1)

    B = np.roll(B, B_offset_X, axis=0)
    B = np.roll(B, B_offset_Y, axis=1)

    print("G_offset: ",(G_offset_X, G_offset_Y))
    print("B_offset: ", (B_offset_X, B_offset_Y))


    # fix G
    # R_offset_X, R_offset_Y = best_offset(G, R, metric, np.arange(-15,15), np.arange(-15,15))
    # B_offset_X, B_offset_Y = best_offset(G, B, metric, np.arange(-15,15), np.arange(-15,15))

    # R = np.roll(R, R_offset_X, axis=0)
    # R = np.roll(R, R_offset_Y, axis=1)
    # B = np.roll(B, B_offset_X, axis=0)
    # B = np.roll(B, B_offset_Y, axis=1)

    # print("R_offset: ",(R_offset_X, R_offset_Y))
    # print("B_offset: ", (B_offset_X, B_offset_Y))


    #fix B
    # R_offset_X, R_offset_Y = best_offset(B, R, metric, np.arange(-15,15), np.arange(-15,15))
    # G_offset_X, G_offset_Y = best_offset(B, G, metric, np.arange(-15,15), np.arange(-15,15))

    # R = np.roll(R, R_offset_X, axis=0)
    # R = np.roll(R, R_offset_Y, axis=1)

    # G = np.roll(G, G_offset_X, axis=0)
    # G = np.roll(G, G_offset_Y, axis=1)

    # print("R_offset: ",(R_offset_X, R_offset_Y))
    # print("G_offset: ", (G_offset_X, G_offset_Y))



    aligned_image = np.stack((R,G,B), axis=2)

    return aligned_image


def pyramid_align(R, G, B, metric):
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively


    pass


def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it 
    
    #triptych = plt.imread('prokudin-gorskii/00125v.jpg')
    #triptych = plt.imread('prokudin-gorskii/00149v.jpg')
    #triptych = plt.imread('prokudin-gorskii/00153v.jpg')
    #triptych = plt.imread('prokudin-gorskii/00351v.jpg')
    #triptych = plt.imread('prokudin-gorskii/00398v.jpg')
    #triptych = plt.imread('prokudin-gorskii/01112v.jpg')
    triptych = plt.imread('tableau/efros_tableau.jpg')
    R, G, B = split_triptych(triptych)
    colored_image = np.stack((R,G,B), axis=2)
    plt.imsave('colored.png',colored_image)
    

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    #best_offset(R, G, normalized_cross_correlation, np.arange(-15,15), np.arange(-15,15))
    aligned_image = align_and_combine(R,G,B,normalized_cross_correlation)
    plt.imsave('aligned_colored_6.png', aligned_image)
    nonnormalized_image = align_and_combine(R,G,B,cross_correlation)
    plt.imsave('nonnormalized_aligned_6.png', nonnormalized_image)
    
    # Task 3: Pyramid alignment


    pass


def part3():
    # TODO: Solution for Q3
    pass
    

def main():
    #part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
