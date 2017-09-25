"""
Implementation of t-SNE based on Van Der Maaten and Hinton (2008)
http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.spatial import distance_matrix
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
    # We take an exp later. We want diagonal entries after
    # this exp to be = 0. Thus set them to -inf
    np.fill_diagonal(D, np.inf)
    return -D


def softmax(X):
    """Compute softmax values for each row of matrix X."""
    # Subtract max for numerical stability
    e_x = np.exp(X - np.max(X, axis=1).reshape([-1, 1]))
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


def calc_perplexity(prob_matrix):
    """Calculate the perplexity of each row of a matrix of probabilities."""
    entropy = -np.sum(prob_matrix * np.log2(prob_matrix), 1)
    perplexity = 2 ** entropy
    return perplexity


def perplex(distances, sigmas):
    """Wrapper function for quick calculation of perplexity over a matrix."""
    return calc_perplexity(calc_prob_matrix(distances, sigmas))


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


def find_optimal_sigmas(distances, target_perplexity, sigma_init=1.0):
    """Find optimal sigmas for each row of distances matrix."""
    sigmas = []
    # For each row of the matrix (each point in our dataset)
    for i in range(distances.shape[0]):
        # Want to find the sigma that gives desired perplexity for this row
        eval_fn = lambda sigma: perplex(distances[i:i+1, :], np.array(sigma))
        # Do binary search over input sigmas to eval_fn to achieve target perp
        correct_sigma = binary_search(eval_fn, target_perplexity)
        # Append the resulting sigma to our output array
        sigmas.append(correct_sigma)
    return np.array(sigmas)


NUM_POINTS = 500
CLASSES_TO_PLOT = [0, 1]
PERPLEXITY = 5
SEED = 1
MOMENTUM = 0.0
ETA = 0.1

# numpy RandomState for reproducability
rng = np.random.RandomState(SEED)

# Load the first 500 0's and 1's from the MNIST dataset
X, y = load_mnist('/home/liam/datasets/', digits_to_keep=CLASSES_TO_PLOT,
                  N=NUM_POINTS)

# =============================================================================
# D = 100
# X = rng.normal(size=[NUM_POINTS, D]) / 10.
# X[:NUM_POINTS//2, :D//2] = X[:NUM_POINTS//2, :D//2] + 0.5
# X[NUM_POINTS//2:, D//2:] = X[NUM_POINTS//2:, D//2:] + 0.5
# y = np.zeros(X.shape[0])
# y[NUM_POINTS//2:] = 1
# =============================================================================

# Get the negative euclidian distance from every data point to all others
#distances = -np.square(distance_matrix(X, X))
distances = neg_squared_euc_dists(X)
# Find optimal sigma for each row of this matrix, given our desired perplexity
sigmas = find_optimal_sigmas(distances, target_perplexity=PERPLEXITY)
# Calculate the probabilities based on these optimal sigmas
p_matrix = calc_prob_matrix(distances, sigmas)
#print(calc_perplexity(p_matrix))
#%%
def Hbeta(D = np.array([]), beta = 1.0):
    """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta);
    sumP = sum(P);
    H = np.log(sumP) + beta * np.sum(D * P) / sumP;
    P = P / sumP;
    return H, P;

def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
    """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""

    # Initialize some variables
    (n, d) = X.shape;
    sum_X = np.sum(np.square(X), 1);
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
    P = np.zeros((n, n));
    beta = np.ones((n, 1));
    logU = np.log(perplexity);

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf;
        betamax =  np.inf;
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
        (H, thisP) = Hbeta(Di, beta[i]);

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU;
        tries = 0;
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy();
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2;
                else:
                    beta[i] = (beta[i] + betamax) / 2;
            else:
                betamax = beta[i].copy();
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2;
                else:
                    beta[i] = (beta[i] + betamin) / 2;

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i]);
            Hdiff = H - logU;
            tries = tries + 1;

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;

    # Return final P-matrix
    print("Mean value of sigma: ", np.mean(np.sqrt(1 / beta)))
    return P, beta;

p2, sig2 = x2p(X, perplexity=5)
#%%
def categorical_scatter_2d(X2D, class_idxs, ms=3, ax=None, 
                           alpha=0.1, legend=False):
    try:
        __IPYTHON__
    except NameError:
        return None

    ## Plot a 2D matrix with corresponding class labels: each class diff colour
    if ax is None:
        fig, ax = plt.subplots()
    #ax.margins(0.05) # Optional, just adds 5% padding to the autoscaling
    classes = [0, 1]
    colors = np.array([[ 0.5742 ,  0.76725,  0.999  ],
                       [ 0.999  ,  0.52965,  0.4356 ]])
    markers = 'os'
    for i, cls in enumerate(classes):
        mark = markers[i]
        ax.plot(X2D[class_idxs==cls, 0], X2D[class_idxs==cls, 1], marker=mark, 
            linestyle='', ms=ms, label=str(cls), alpha=alpha, color=colors[i],
            markeredgecolor='black', markeredgewidth=0.4)
    if legend:
        ax.legend()
    return ax

#%%

def pca(X = np.array([]), no_dims = 2):
	"""Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""

	print("Preprocessing the data using PCA...")
	(n, d) = X.shape;
	X = X - np.tile(np.mean(X, 0), (n, 1));
	(l, M) = np.linalg.eig(np.dot(X.T, X));
	Y = np.dot(X, M[:,0:no_dims]);
	return Y;

categorical_scatter_2d(pca(X, 2).real, y, alpha=1)
#%%
# Initialise our 2D representation
low_dim_X = rng.normal(0., 0.5, [X.shape[0], 2]).round(2)
#low_dim_X = pca(X, 2).real
low_dim_X_m2 = low_dim_X
low_dim_X_m1 = low_dim_X

num_iters = 5000
def stdev(i, num_iters):
    return (0.5 - np.tanh(8. * i * 1. / num_iters - 4) / 2) * 0.05

for i in range(num_iters):
    q_matrix = softmax(neg_squared_euc_dists(low_dim_X))
    
    pp = p_matrix + p_matrix.T
    qq = q_matrix + q_matrix.T
    
    diff = (pp - qq).round(2)
    
    res = (np.expand_dims(low_dim_X, 1) - np.expand_dims(low_dim_X, 0))
    
    res = res.round(2)
    
    grads = (np.expand_dims(diff, 2) * res).sum(1)
    
    low_dim_X = low_dim_X - ETA * grads
    low_dim_X = low_dim_X + rng.normal(0., stdev(i, num_iters), low_dim_X.shape)
    
    if i % (num_iters / 10) == 0:
        categorical_scatter_2d(low_dim_X, y, alpha=1.0, ms=10)
        plt.show()
    
    #%%
    diff = p_matrix - q_matrix
    symm = diff + diff.T
    res = (np.expand_dims(low_dim_X, 1) - np.expand_dims(low_dim_X, 0))
    grads = -2 * (np.expand_dims(symm, 2) * res).sum(1)
    
    prev_change = low_dim_X_m1 - low_dim_X_m2
    low_dim_X = low_dim_X + ETA * grads# + MOMENTUM * prev_change - 0.001 * low_dim_X
    low_dim_X = low_dim_X# + rng.normal(0., stdev(i, num_iters), low_dim_X.shape)
    low_dim_X_m2 = low_dim_X_m1.copy()
    low_dim_X_m1 = low_dim_X.copy()
    if i % (num_iters / 10) == 0:
        categorical_scatter_2d(low_dim_X, y, alpha=1.0)
        plt.show()