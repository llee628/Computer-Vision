import numpy as np
from numpy import linalg as LA
import pdb


def affine_transform_loss(P, P_prime, S, t):
    """
    Compute loss, predictions, and gradients for a set of 2D correspondences
    and a candidate affine transform to align them.

    More specifically, use the affine transform (S, t) to transform the points
    P into predictions. The loss is equal to the average squared L2 distance
    between predictions and points in P_prime.

    Inputs:
    - P, P_prime: Numpy arrays of shape (N, 2) giving correspondences.
    - S, t: Numpy arrays of shape (2, 2) and (2,) giving parameters of an
      affine transform.

    Returns a tuple of:
    - loss: A float giving the loss
    - prediction: Numpy array of shape (N, 2) giving predicted points, where
      prediction[i] is the result of applying the affine transform (S, t) to
      the input point P[i].
    - grad_S: Numpy array of shape (2, 2) giving the gradient of the loss with
      respect to the affine transform parameters S.
    - grad_t: Numpy array of shape (2,) giving the gradient of the loss with
      respect to the affine transform parameters t.
    """
    # Forward pass: Compute loss and prediction
    loss = None
    prediction = None
    ###########################################################################
    # TODO: Implement the forward pass to compute the predictions and loss,   #
    # storing them in the variables above. Your implementation should be      #
    # fully vectorized, and should not contain any loops (including map,      #
    # filter, or comprehension expressions).                                  #
    ###########################################################################
    N = P.shape[0]
    #breakpoint()
    # prediction = S.dot(P.T) + t.reshape(t.shape[0], 1)
    # prediction = prediction.T
    # loss = LA.norm((prediction - P_prime), axis=1)
    # loss = (np.sum(loss**2))/N
    #breakpoint()
    P = P.T
    a = S.dot(P)
    b = a + t.reshape(t.shape[0],1)
    prediction = b.T
    c = P_prime - prediction
    d = LA.norm(c, axis=1)
    d = d**2
    e = np.sum(d)
    loss = e/N
    #breakpoint()
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    # Backward pass: Compute gradients
    grad_S, grad_t = None, None
    ###########################################################################
    # TODO: Implement the backward pass to compute the gradient of the loss   #
    # with respect to the transform parameters S and t. Store the gradients   #
    # in the grad variables defined above. As above, your implementation      #
    # should be fully vectorized and should not contain any loops.            #
    ###########################################################################
    grad_L = 1.0
    grad_e = (1.0/N)*grad_L
    grad_d = np.ones((N,1))*grad_e
    grad_c = 2*c*grad_d
    grad_predict = -1.0*grad_c
    grad_b = grad_predict.T
    grad_t = grad_b.dot(np.ones((N,1)))
    grad_t = np.squeeze(grad_t)
    grad_a = 1.0*grad_b
    grad_S = grad_a.dot(P.T)
    #breakpoint()


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return loss, prediction, grad_S, grad_t


def fit_affine_transform(P, P_prime, logger, learning_rate, steps):
    """
    Use gradient descent to fit an affine transform to a set of corresponding
    points. The transformation should be initialized to the identity transform.

    This function also takes a Logger object as an additional argument which
    can help to track the fitting process. Each iteration of gradient descent
    should include a call of the form:

    logger.log(i, loss, prediction)

    where i is an integer giving the current iteration, and loss and prediction
    give outputs from affine_transform_loss at the current iteration.

    Inputs:
    - P, P_prime: Numpy arrays of shape (N, 2) giving 2D correspondences.
    - logger: A Logger object (see above)
    - learning_rate: The learning rate to use for gradient descent updates
    - steps: The number of iterations of gradient descent to use

    Returns a tuple giving parameters of the affine transform fit to the data:
    - S: Numpy array of shape (2, 2)
    - t: Numpy array of shape (2,)
    """
    S, t = None, None
    ###########################################################################
    # TODO: Use gradient descent to fit an affine transform to the data,      #
    # storing the parameters of the transform in the variables above.         #
    # Don't forget to call the logger at each iteration!                      #
    ###########################################################################
    S = np.array([[1, 0],[0, 1]])
    t = np.array([0, 0])
    #breakpoint()
    for i in range(steps):
        loss, prediction, grad_S, grad_t = affine_transform_loss(P, P_prime, S, t)
        logger.log(i, loss, prediction)
        S = S - grad_S*learning_rate
        t = t - grad_t*learning_rate
    #breakpoint()

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return S, t
