"""
Load MNIST dataset.

Author: Liam Schoneveld
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from six.moves import cPickle as pickle
from six.moves import urllib
import numpy as np
import gzip

DEFAULT_SOURCE_URL = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'


def _make_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _maybe_download(filename, directory, source_url):
    _make_dir_if_not_exist(directory)
    filepath = os.path.join(directory, filename)
    if not os.path.exists(filepath):
        print('Downloading', filename, '...')
        filepath, _ = urllib.request.urlretrieve(source_url, filepath)
        print('Successfully downloaded', filename)
    return filepath


def load_mnist(datasets_path, digits_to_keep=[0, 1], N=500):
    
    # Download the dataset if we don't have it already
    path = _maybe_download('mnist.pkl.gz', datasets_path, DEFAULT_SOURCE_URL)
    
    # Load the dataset
    f = gzip.open(path, 'rb')
    train_set, _, _ = pickle.load(f)
    f.close()
    
    # Find indices of digits in training set that we will keep
    includes_matrix = [(train_set[1]==i) for i in digits_to_keep]
    keep_indices = np.sum(includes_matrix, 0).astype(np.bool)
    
    # Drop images of any other digits
    train_set = [train_set[0][keep_indices], train_set[1][keep_indices]]
    
    # Only keep the first N examples
    N = min(N, train_set[0].shape[0])
    train_set = [train_set[0][:N], train_set[1][:N]]
    
    return train_set
