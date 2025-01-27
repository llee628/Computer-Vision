import os

import numpy as np
import matplotlib.pyplot as plt

from common import read_img, save_img
import pdb


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    # TODO: Use slicing to complete the function
    #breakpoint()
    output = []
    H = image.shape[0]
    W = image.shape[1]
    M = patch_size[0]
    N = patch_size[1]

    for r in range(0, H - M):
        for c in range(0, W - N):
            patch = image[r:r+M, c:c+N]
            Mean = np.mean(patch)
            Std = np.std(patch)
            patch = (patch - Mean)/Std
            #breakpoint()
            output.append(patch)

    #breakpoint()    

    return output


def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """

    H = image.shape[0]
    W = image.shape[1]
    h = kernel.shape[0]
    w = kernel.shape[1]

    output = np.zeros((H,W))

    #flip the order of the kernel
    kernel = kernel[::-1, ::-1]

    #zero-padding
    vertical_size = h//2
    horizon_size = w//2
    image = np.pad(image, ((vertical_size,vertical_size),(horizon_size,horizon_size)), 'constant')

    #convolution
    for r in range(H):
        for c in range(W):
            output[r,c] = np.sum(image[r:r+h, c:c+w]*kernel)

    #breakpoint()

    return output


def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # TODO: Fix kx, ky
    kx = np.array([[1, 0, -1]])  # 1 x 3
    ky = np.array([[1],[0],[-1]])  # 3 x 1


    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

   
    # TODO: Use Ix, Iy to calculate grad_magnitude
    grad_magnitude = np.sqrt((Ix**2) + (Iy**2) )

    #breakpoint()
    return Ix, Iy, grad_magnitude


def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # TODO: Use convolve() to complete the function
    #Gx, Gy, grad_magnitude = None, None, None
    Sx = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
    Sy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    
    Gx = convolve(image, Sx)
    Gy = convolve(image, Sy)
    grad_magnitude = np.sqrt((Gx**2) + (Gy**2))
    #breakpoint()

    return Gx, Gy, grad_magnitude

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



def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    #(a)
    #First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = patches[0]
    chosen_patches = np.vstack((chosen_patches, patches[1]))
    chosen_patches = np.vstack((chosen_patches, patches[2]))
    save_img(chosen_patches, "./image_patches/q1_patch.png")
    
    #(b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    #breakpoint()

    # -- TODO Task 2: Convolution and Gaussian Filter --
    
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    sigma = 0.572
    #sigma = 2.0
    kernel_gaussian = gaussian_kernel_generator(sigma)


    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")
    #breakpoint()

    #
    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the orignal and gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")
    #breakpoint()
    

    # -- TODO Task 3: Sobel Operator --
    
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    #breakpoint()
    
    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])

    #breakpoint()
    
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    data = np.load('log1d.npz')
    LoG_50 = data['log50']
    gauss_50 = data['gauss50']
    gauss_53 = data['gauss53']
    DoG = gauss_53 - gauss_50
    x = np.arange(-250,251)
    plt.plot(x, LoG_50, label='Laplacian')
    plt.plot(x, DoG, label='DoG')
    plt.legend()
    plt.show()
    #breakpoint()
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
