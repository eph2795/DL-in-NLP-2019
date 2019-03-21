import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    N = X.shape[0]
   
    z = X.dot(W)
    z_max = z.max(axis=1, keepdims=True)
    z = z - z_max
    exp = np.exp(z)
    exp_sum = exp.sum(axis=1, keepdims=True)
    soft_max = exp / exp_sum
    
    for i in range(N):
        loss += -1 / N * (z[i, y[i]] - np.log(exp_sum[i]))
    loss += reg / (2 * N) * np.square(W).sum()
        
    dl_dz = soft_max / N
    for i in range(N):
        dl_dz[i, y[i]] -= 1 / N
        
    dW = X.T.dot(dl_dz) + reg / N * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    N = X.shape[0]
    
    z = X.dot(W)
    z_max = z.max(axis=1, keepdims=True)
    z = z - z_max
    exp = np.exp(z)
    exp_sum = exp.sum(axis=1, keepdims=True)
    soft_max = exp / exp_sum
    true_indices = (np.arange(y.shape[0]), y) 
 
    loss = 1 / N * (-z[true_indices] + np.log(exp_sum.flatten())).sum() + reg / (2 * N) * np.square(W).sum()
    
    dl_dz = soft_max / N
    dl_dz[true_indices] -= 1 / N
    dW = X.T.dot(dl_dz) + reg / N * W
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
