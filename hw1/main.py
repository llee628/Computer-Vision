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
    
    #task1_1()



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
    R, G, B = None, None, None
    # TODO: Split a triptych into thirds and 
    # return three channels as numpy arrays
    return R, G, B


def normalized_cross_correlation(ch1, ch2):
    """
    Calculates similarity between 2 color channels
    Input:  ch1: channel 1 matrix
            ch2: channel 2 matrix
    Output: normalized cross correlation (scalar)
    """
    pass


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
    pass


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
    pass


def pyramid_align():
    # TODO: Reuse the functions from task 2 to perform the 
    # image pyramid alignment iteratively or recursively
    pass


def part2():
    # TODO: Solution for Q2
    # Task 1: Generate a colour image by splitting 
    # the triptych image and save it 

    # Task 2: Remove misalignment in the colour channels 
    # by calculating best offset
    
    # Task 3: Pyramid alignment
    pass


def part3():
    # TODO: Solution for Q3
    pass
    

def main():
    part1()
    part2()
    part3()


if __name__ == "__main__":
    main()
