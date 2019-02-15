import numpy as np
from random import shuffle
import math

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
  num_classes = W.shape[1] 
  num_train = X.shape[0]

  for i in range(num_train):
    scores = X[i].dot(W)
    exps = np.exp(scores)
    prob = exps[y[i]] / np.sum(exps)
    loss += -math.log(prob)
    for j in range(num_classes):
      dW[:, j] += X[i] * (exps[j] / np.sum(exps))
      if (j == y[i]):
          dW[:, j] -= X[i]

  dW /= num_train
  dW += reg * 2 * W
  loss /= num_train
  loss += reg * np.sum(W * W)

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_classes = W.shape[1]

  #loss at first
  scores = np.dot(X, W)
  exps = np.exp(scores)
  exps_yi = exps[np.arange(num_train), y]
  sums = np.sum(exps, axis=1)
  losses = -np.log(exps_yi / sums)
  loss = np.sum(losses)
  loss /= num_train
  loss += reg * np.sum(W * W)

  # gradient then
  A = exps / sums.reshape(-1, 1)
  A[np.arange(num_train), y] -= 1
  dW = np.dot(X.T, A) 
  dW /= num_train 
  dW += reg * 2 * W

  return loss, dW

