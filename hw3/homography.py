"""
Homography fitting functions
You should write these
"""
import numpy as np
import matplotlib.pyplot as plt
import pdb
from common import homography_transform

def fit_homography(XY):
    '''
    Given a set of N correspondences XY of the form [x,y,x',y'],
    fit a homography from [x,y,1] to [x',y',1].
    
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
    Output -H: a (3,3) homography matrix that (if the correspondences can be
            described by a homography) satisfies [x',y',1]^T === H [x,y,1]^T

    '''
    #breakpoint()
    N = XY.shape[0]
    p = np.ones((N,3))
    p_comma = np.ones((N,3))

    p[:,0] = XY[:,0]
    p[:,1] = XY[:,1]
    p_comma[:,0] = XY[:,2]
    p_comma[:,1] = XY[:,3]
    x = p_comma[:, [0]]
    y = p_comma[:, [1]]
    
    A = np.zeros((2*N,9))
    A[0:N, 3:6] = -1*p
    A[0:N, 6:] = y*p
    A[N:, 0:3] = p
    A[N:, 6:] = -1*x*p
    
    A_TA = np.dot(A.T,A)
    w,v = np.linalg.eig(A_TA)
    h = v[:,np.argmin(w)]
    h = h/h[8]
    H = h.reshape(3,3)
    #breakpoint()

    return H


def RANSAC_fit_homography(XY, eps=1, nIters=1000):
    '''
    Perform RANSAC to find the homography transformation 
    matrix which has the most inliers
        
    Input - XY: an array with size(N,4), each row contains two
            points in the form [x_i, y_i, x'_i, y'_i] (1,4)
            eps: threshold distance for inlier calculation
            nIters: number of iteration for running RANSAC
    Output - bestH: a (3,3) homography matrix fit to the 
                    inliers from the best model.

    Hints:
    a) Sample without replacement. Otherwise you risk picking a set of points
       that have a duplicate.
    b) *Re-fit* the homography after you have found the best inliers
    '''
    bestH, bestCount, bestInliers = np.eye(3), -1, np.zeros((XY.shape[0],))
    bestRefit = np.eye(3)
    return bestRefit

def get_p(data):
    N = data.shape[0]
    x = data[:,0]
    y = data[:,1]

    p = np.ones((N,3))
    p[:,0] = x
    p[:,1] = y
    return p


def part_b():
    data_case1 = np.load("./task4/points_case_1.npy")
    data_case4 = np.load("./task4/points_case_4.npy")
    
    H = fit_homography(data_case1)
    print("H1 =")
    print(H)
    print("")

    H = fit_homography(data_case4)
    print("H4 =")
    print(H)

def part_c():
    data_case5 = np.load("./task4/points_case_5.npy")
    data_case9 = np.load("./task4/points_case_9.npy")

    H = fit_homography(data_case5)
    p = get_p(data_case5)
    homo_tran = np.dot(H, p.T)
    homo_tran = homo_tran.T

    plt.scatter(data_case5[:,0], data_case5[:,1], 1, c="red", label="[x, y]")
    plt.scatter(data_case5[:,2], data_case5[:,3], 10, c="green", label="[x', y']")
    plt.scatter(homo_tran[:,0], homo_tran[:,1], 1, c="blue", label="T (H, [xi, yi])")
    plt.legend()
    plt.savefig("4_c_1.png")
    plt.close()
    #plt.show()

    H = fit_homography(data_case9)
    p = get_p(data_case9)
    homo_tran = np.dot(H, p.T)
    homo_tran = homo_tran.T

    plt.scatter(data_case9[:,0], data_case9[:,1], 1, c="red", label="[x, y]")
    plt.scatter(data_case9[:,2], data_case9[:,3], 10, c="green", label="[x', y']")
    plt.scatter(homo_tran[:,0], homo_tran[:,1], 1, c="blue", label="T (H, [xi, yi])")
    plt.legend()
    plt.savefig("4_c_2.png")
    #plt.show()
    #plt.close()





if __name__ == "__main__":
    #If you want to test your homography, you may want write any code here, safely
    #enclosed by a if __name__ == "__main__": . This will ensure that if you import
    #the code, you don't run your test code too
    part_b()
    part_c()
    pass
