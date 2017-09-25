"""
Implementation of t-SNE based on Van Der Maaten and Hinton (2008)
http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from load_data import load_mnist
import matplotlib.pyplot as plt


def neg_squared_euc_dists(X):
    """Compute matrix containing negative squared euclidean 
    distance for all pairs of points in input matrix X
    
    # Arguments:
        X: matrix of size NxD
    # Returns:
        NxN matrix D, with entry D_ij = negative squared
        euclidean distance between rows X_i and X_j
    """
    # Math? See https://stackoverflow.com/questions/37009647
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    return -D


def softmax(X, diag_zero=True):
    """Compute softmax values for each row of matrix X."""
    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
    # We usually want diagonal probailities to be 0.
    if diag_zero:
        np.fill_diagonal(e_x, 0.)
    # Add a tiny constant for stability of log we take later
    e_x = e_x + 1e-8  # numerical stability
    return e_x / e_x.sum(axis=1).reshape([-1, 1])


def calc_prob_matrix(distances, sigmas=None):
    """Convert a distances matrix to a matrix of probabilities."""
    if sigmas is not None:
        two_sig_sq = 2. * np.square(sigmas.reshape((-1, 1)))
        return softmax(distances / two_sig_sq)
    else:
        return softmax(distances)


def binary_search(eval_fn, target, tol=1e-10, max_iter=10000, 
                  lower=1e-20, upper=1000.):
    """Perform a binary search over input values to eval_fn.
    
    # Arguments
        eval_fn: Function that we are optimising over.
        target: Target value we want the function to output.
        tol: Float, once our guess is this close to target, stop.
        max_iter: Integer, maximum num. iterations to search for.
        lower: Float, lower bound of search range.
        upper: Float, upper bound of search range.
    # Returns:
        Float, best input value to function found during search.
    """
    for i in range(max_iter):
        guess = (lower + upper) / 2.
        val = eval_fn(guess)
        if val > target:
            upper = guess
        else:
            lower = guess
        if np.abs(val - target) <= tol:
            break
    return guess


def calc_perplexity(prob_matrix):
    """Calculate the perplexity of each row 
    of a matrix of probabilities."""
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplexity(distances, sigmas):
    """Wrapper function for quick calculation of 
    perplexity over a distance matrix."""
    return calc_perplexity(calc_prob_matrix(distances, sigmas))


def find_optimal_sigmas(distances, target_perplexity):
    """For each row of distances matrix, find sigma that results
    in target perplexity for that role."""
    sigmas = [] 
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Make fn that returns perplexity of this row given sigma
        eval_fn = lambda sigma: \
            perplexity(distances[i:i+1, :], np.array(sigma))
        # Binary search over sigmas to achieve target perplexity
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


def categorical_scatter_2d(X2D, class_idxs, ms=3, ax=None, 
                           alpha=0.1, legend=False):
    ## Plot a 2D matrix with corresponding class labels: each class diff colour
    if ax is None:
        fig, ax = plt.subplots()
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    classes = list(np.unique(class_idxs))
    markers = 'os' * 10
    colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(X2D[class_idxs==cls, 0], X2D[class_idxs==cls, 1], marker=mark, 
            linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
            markeredgecolor='black', markeredgewidth=0.4)
    if legend:
        ax.legend()
    return ax


def pca(X = np.array([]), no_dims = 2):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;


def q_tsne(Y):
    """Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = neg_squared_euc_dists(Y)
    inv_distances = np.power(1. - distances, -1)
    np.fill_diagonal(inv_distances, 0.)
    return inv_distances / np.sum(inv_distances), inv_distances


def q_joint(Y):
    """Given low-dimensional representations Y, compute
    matrix of joint probabilities with entries q_ij."""
    distances = neg_squared_euc_dists(Y)
    exp_distances = np.exp(distances)
    np.fill_diagonal(exp_distances, 0.)
    return exp_distances / np.sum(exp_distances), None


def p_joint(P):
    """Given conditional probabilities matrix P, return
    approximation of joint distribution probabilities."""
    return (P + P.T) / (2. * P.shape[0])

    
def stdev(i, num_iters):
    return (0.5 - np.tanh(8. * i * 1. / num_iters - 4) / 2)

def symmetric_sne_grad(P, Q, Y, _):
    pq_diff = P - Q
    y_diffs = (np.expand_dims(Y, 1) - np.expand_dims(Y, 0))
    grad = 4. * (np.expand_dims(pq_diff, 2) * y_diffs).sum(1)
    return grad

def tsne_grad(P, Q, Y, distances):
    pq_diff = P - Q
    y_diffs = (np.expand_dims(Y, 1) - np.expand_dims(Y, 0))
    y_diffs_wt_dist = y_diffs * np.expand_dims(distances, 2)
    grad = 4. * (np.expand_dims(pq_diff, 2) * y_diffs_wt_dist).sum(1)
    return grad
    
def estimate_sne(X,             # Input data matrix
                 P,             # Matrix of join probabilities
                 rng,           # np.random.RandomState()
                 num_iters,     # Iterations to train for
                 q_fn,          # Function from Y->q join probs
                 grad_fn,       # Function for estimating gradient
                 pca_init,      # Whether to initialise Y with PCA
                 learning_rate, # 
                 momentum, 
                 annealing, 
                 plot=True):
    
    # Initialise our 2D representation
    if pca_init:
        Y = pca(X, 2).real
    else:
        Y = rng.normal(0., 0.5, [X.shape[0], 2]).round(2)
    
    # Initialise past values (used for momentum)
    if momentum:
        Y_m2 = Y.copy()
        Y_m1 = Y.copy()
    
    # Start gradient descent loop
    for i in range(num_iters):
        
        Q, distances = q_fn(Y)
        grads = grad_fn(P, Q, Y, distances)
        
        Y = Y - learning_rate * grads
        if momentum:
            Y += momentum * (Y_m1 - Y_m2)
        if annealing:
            Y += rng.normal(
                    0., stdev(i, num_iters) * annealing, Y.shape)
        
        Y_m1 = Y.copy()
        Y_m2 = Y_m1.copy()
        
        if plot and i % (num_iters / 10) == 0:
            categorical_scatter_2d(Y, y, alpha=1.0, ms=10)
            plt.show()

    return Y

def main():
    
    # Set global parameters
    NUM_POINTS = 200
    CLASSES_TO_USE = [0, 1, 2, 3]  # MNIST classes to use
    PERPLEXITY = 20
    SEED = 1
    MOMENTUM = 0.9
    ETA = 10.
    NUM_ITERS = 2000
    ANNEALING = 0.
    PCA_INIT = False
    
    # numpy RandomState for reproducability
    rng = np.random.RandomState(SEED)
    
    # Load the first 500 0's and 1's from the MNIST dataset
    X, y = load_mnist('/home/liam/datasets/',
                      digits_to_keep=CLASSES_TO_USE,
                      N=NUM_POINTS)
    # Get the negative euclidian distances matrix for our data
    distances = neg_squared_euc_dists(X)
    # Find optimal sigma for each row of this matrix, given our desired perplexity
    sigmas = find_optimal_sigmas(distances, target_perplexity=PERPLEXITY)
    # Calculate the probabilities based on these optimal sigmas
    p_conditional = calc_prob_matrix(distances, sigmas)
    #print(calc_perplexity(p_matrix))
    
    P = p_joint(p_conditional)
    estimate_sne(P,
                 num_iters=NUM_ITERS,
                 q_fn=q_fn,
                 grad_fn=grad_fn,
                 pca_init=PCA_INIT,
                 momentum=MOMENTUM, 
                 annealing=ANNEALING, 
                 plot=True)
        
main()