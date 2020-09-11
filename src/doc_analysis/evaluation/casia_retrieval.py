"""
Created on June 13, 2020

@author: yhe
"""
import logging
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


def kws(query_features, test_features,
        query_labels, test_labels,
        metric,
        drop_first=False):
    logging.info('---Running Keyword Spotting---')

    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')
    else:
        # compute the nearest neighbors
        dist_mat = cdist(XA=query_features, XB=test_features, metric='cosine')
        retrieval_indices = np.argsort(dist_mat, axis=1)

    # create the retrieval matrix
    retr_mat = np.tile(test_labels, (len(query_labels), 1))
    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels), 1)))
    retr_mat = retr_mat[row_selector, retrieval_indices]

    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    mean_ap = np.mean(avg_precs)
    return mean_ap, avg_precs



if __name__ == '__main__':
    pass
