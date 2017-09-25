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

def softmax(x, axis=1):
    """Compute softmax values for each row of matrix x."""
    rshp = [-1, 1] if axis == 1 else [1, -1]
    e_x = np.exp(x - np.max(x, axis=axis).reshape(rshp))
    return e_x / e_x.sum(axis=axis).reshape(rshp)


def calc_prob_matrix(distances, sigmas):
    """Convert a distance matrix to a matrix of probabilities."""
    dists_div_sigmas = distances / (2. * np.square(sigmas.reshape((-1, 1))))
    prob_matrix = softmax(dists_div_sigmas)
    return prob_matrix


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
        tol: Float, once our guess is this close or less to target, stop.
        max_iter: Integer, maximum number of iterations to search for.
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
        eval_fn = lambda sigma: perplex(distances[i:i+1, :], np.array(sigma))
        correct_sigma = binary_search(eval_fn, target_perplexity)
        sigmas.append(correct_sigma)
    return np.array(sigmas)


NUM_POINTS = 3
CLASSES_TO_PLOT = [0, 1]
PERPLEXITY = 1.5
SEED = 1

# numpy RandomState for reproducability
rng = np.random.RandomState(SEED)

# Load the first 500 0's and 1's from the MNIST dataset
X, y = load_mnist('/home/liam/datasets/', digits_to_keep=CLASSES_TO_PLOT,
                  N=NUM_POINTS)

# Get the negative euclidian distance from every data point to all others
distances = -np.square(distance_matrix(X, X))
# Find optimal sigma for each row of this matrix, given our desired perplexity
sigmas = find_optimal_sigmas(distances, target_perplexity=PERPLEXITY)
# Calculate the probabilities based on these optimal sigmas
p_matrix = calc_prob_matrix(distances, sigmas)
#print(calc_perplexity(p_matrix))

# Initialise our 2D representation
low_dim_X = rng.normal(0., 0.05, [X.shape[0], 2])
#%%
q_matrix = softmax(-np.square(distance_matrix(low_dim_X, low_dim_X)))

#%%
def neg_euc_dist(i, j, X=X):
    return -np.sum(np.square(X[i:i+1] - X[j:j+1]))
def scaled_sig(i):
    return 2 * (sigmas[i] ** 2)

out = {}
for j_given_i in [True, False]:
    out[j_given_i] = np.zeros((NUM_POINTS, NUM_POINTS))
    for i in range(NUM_POINTS):
        for j in range(NUM_POINTS):
            div = 0.
            if j_given_i:
                for k in range(NUM_POINTS):
                    d_divd = neg_euc_dist(i, k) / scaled_sig(i)
                    div += np.exp(d_divd)
                out[j_given_i][i,j] = (np.exp(neg_euc_dist(i, j) / scaled_sig(i)) / div)
            else:
                for k in range(NUM_POINTS):
                    d_divd = neg_euc_dist(j, k) / scaled_sig(j)
                    div += np.exp(d_divd)
                out[j_given_i][i,j] = (np.exp(neg_euc_dist(j, i) / scaled_sig(j)) / div)
print(out[True])
print(out[False])


#%%

diff = p_matrix - q_matrix
symm = diff + diff.T

#%%
res = (np.expand_dims(low_dim_X, 0) - np.expand_dims(low_dim_X, 1))
res
#%%
z = low_dim_X
#res = (np.expand_dims(z, 0) - np.expand_dims(z, 1))
res = (np.expand_dims(z, 1) - np.expand_dims(z, 0))
print (np.expand_dims(symm, 2) * res).sum(1)
i = 0
out = []
for i in range(NUM_POINTS):
    s = 0.
    for j in range(NUM_POINTS):
        s += (p_matrix[i, j] - q_matrix[i, j] + p_matrix[j, i] - q_matrix[j, i]) * (z[i] - z[j])
    out.append(s)
print(np.array(out))
