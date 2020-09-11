"""
Created on Dec 1, 2014

@author: ssudholt
"""
import copy
from builtins import print

import numpy as np
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist, pdist, squareform


def average_precision(ret_vec_relevance, gt_relevance_num=None):
    """
    Computes the average precision from a list of relevance items

    Params:
        ret_vec_relevance: A 1-D numpy array containing ground truth (gt)
            relevance values
        gt_relevance_num: Number of relevant items in the data set
            (with respect to the ground truth)
            If None, the average precision is calculated wrt the number of
            relevant items in the retrieval list (ret_vec_relevance)

    Returns:
        The average precision for the given relevance vector.
    """
    if ret_vec_relevance.ndim != 1:
        raise ValueError('Invalid ret_vec_relevance shape')

    ret_vec_cumsum = np.cumsum(ret_vec_relevance, dtype=float)
    ret_vec_range = np.arange(1, ret_vec_relevance.size + 1)
    ret_vec_precision = ret_vec_cumsum / ret_vec_range

    if gt_relevance_num is None:
        n_relevance = ret_vec_relevance.sum()
    else:
        n_relevance = gt_relevance_num

    if n_relevance > 0:
        ret_vec_ap = (ret_vec_precision * ret_vec_relevance).sum() / n_relevance
    else:
        ret_vec_ap = 0.0
    return ret_vec_ap


def map_from_query_test_feature_matrices(query_features, test_features,
                                         query_labels, test_labels,
                                         metric='cosine',
                                         drop_first=False):
    """
    Compute the mAP for a given list of queries and test instances
    Each query is used to rank the test samples
    :param query_features: (2D ndarray)
        feature representation of the queries
    :param test_features: (2D ndarray)
        feature representation of the test instances
    :param query_labels: (1D ndarray or list)
        the labels corresponding to the queries (either numeric or characters)
    :param test_labels: (1D ndarray or list)
        the labels corresponding to the test instances (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    """
    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')

    # compute the nearest neighbors
    print("<<<<<<<<<<<< query_features")
    print(type(query_features))
    print(query_features.shape)
    print(query_features)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< test_features")
    print(type(test_features))
    print(test_features.shape)
    print(test_features)
    print(">>>>>>><>>>>")

    '''
    XA.shape = [m, n]
    XB.shape = [x, y]
    dist_mat.shape = [m, x]
    '''
    # reduce the dimension of qry_features
    # pca = PCA(n_components=2000)
    # query_features = pca.fit_transform(query_features)

    # if query_features.shape[0] > 5000:
    query_features = query_features[:5000, ]
    query_labels = query_labels[:5000, ]

    # split feature vector which has 5000+ dimensions
    if test_features.shape[0] > 5000:
        # initialize dist_mat: [size_of_query_features, 1]
        dist_mat = np.zeros((query_features.shape[0], 1))
        tmp_test_features = test_features
        size_of_tmp_test_features = tmp_test_features.shape[0]
        while size_of_tmp_test_features > 0:

            if size_of_tmp_test_features > 5000:
                tmp_dist_mat = cdist(XA=query_features, XB=tmp_test_features[:5000, ], metric='cosine')
                tmp_test_features = tmp_test_features[5000:, ]
                size_of_tmp_test_features = tmp_test_features.shape[0]
            else:
                tmp_dist_mat = cdist(XA=query_features, XB=tmp_test_features, metric='cosine')
                tmp_test_features = tmp_test_features[tmp_test_features.shape[0]:, ]
                size_of_tmp_test_features = 0

            # dis_mat: [size_of_query_features, m] + tmp_dis_mat:[size_of_query_features, n]
            # -> dist_mat: [size_of_query_features, m+n]
            # dist_mat = np.concatenate([dist_mat, tmp_dist_mat], axis=1)
            dist_mat = np.hstack((dist_mat, tmp_dist_mat))

        # remove first column after all dist_mat's were concatenated
        dist_mat = np.delete(dist_mat, 0, axis=1)
        print("<<<<<<<<<<<< dist_mat after hstack")
        print(type(dist_mat))
        print(dist_mat.shape)
        print(dist_mat)
        print(">>>>>>><>>>>")
    else:
        dist_mat = cdist(XA=query_features, XB=test_features, metric='cosine')

    retrieval_indices = np.argsort(dist_mat, axis=1)
    print("<<<<<<<<<<<< dist_mat")
    print(type(dist_mat))
    print(dist_mat.shape)
    print(dist_mat)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< retrieval_indices")
    print(type(retrieval_indices))
    print(retrieval_indices.shape)
    print(retrieval_indices)
    print(">>>>>>><>>>>")

    # create the retrieval matrix
    # test_labels.shape = [m, 1]
    retr_mat = np.tile(test_labels, (len(query_labels), 1))

    print("<<<<<<<<<<<< test_labels")
    print(type(test_labels))
    print(test_labels.shape)
    print(test_labels)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< retr_mat")
    print(type(retr_mat))
    print(retr_mat.shape)
    print(retr_mat)
    print(">>>>>>><>>>>")

    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels), 1)))
    print("<<<<<<<<<<<< row_selector")
    print(type(row_selector))
    print(row_selector.shape)
    print(row_selector)
    print(">>>>>>><>>>>")

    retr_mat = retr_mat[row_selector, retrieval_indices]
    print("<<<<<<<<<<<< retr_mat")
    print(type(retr_mat))
    print(retr_mat.shape)
    print(retr_mat)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< qry_labels")
    print(type(query_labels))
    print(query_labels.shape)
    print(query_labels)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< qry_labels.T")
    print(type(np.atleast_2d(query_labels).T))
    print(np.atleast_2d(query_labels).T.shape)
    print(np.atleast_2d(query_labels).T)
    print(">>>>>>><>>>>")

    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    print('<<<<<<<<< avg_precs')
    print(avg_precs)

    mean_ap = np.mean(avg_precs)
    print('<<<<<<<<< mean_ap')
    print(mean_ap)
    return mean_ap, avg_precs


def complete_map_from_qry_test_list(query_features, test_features,
                                    query_labels, test_labels,
                                    metric='cosine',
                                    drop_first=False):
    """
    Compute the mAP for a given list of queries and test instances
    Each query is used to rank the test samples
    :param query_features: (2D ndarray)
        feature representation of the queries
    :param test_features: (2D ndarray)
        feature representation of the test instances
    :param query_labels: (1D ndarray or list)
        the labels corresponding to the queries (either numeric or characters)
    :param test_labels: (1D ndarray or list)
        the labels corresponding to the test instances (either numeric or characters)
    :param metric: (string)
        the metric to be used in calculating the mAP
    :param drop_first: (bool)
        whether to drop the first retrieval result or not
    """
    # some argument error checking
    if query_features.shape[1] != test_features.shape[1]:
        raise ValueError('Shape mismatch')
    if query_features.shape[0] != len(query_labels):
        raise ValueError('The number of query feature vectors and query labels does not match')
    if test_features.shape[0] != len(test_labels):
        raise ValueError('The number of test feature vectors and test labels does not match')

    # compute the nearest neighbors
    print("<<<<<<<<<<<< query_features")
    print(type(query_features))
    print(query_features.shape)
    print(query_features)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< test_features")
    print(type(test_features))
    print(test_features.shape)
    print(test_features)
    print(">>>>>>><>>>>")

    '''
    XA.shape = [m, n]
    XB.shape = [x, y]
    dist_mat.shape = [m, x]
    '''

    # if query_features.shape[0] > 5000:
    query_features = query_features[:5000, ]
    query_labels = query_labels[:5000, ]
    avg_precs = []

    if query_features.shape[0] > 5000:
        # initialize dist_mat: [size_of_query_features, 1]
        tmp_query_features = query_features
        size_of_tmp_query_features = tmp_query_features.shape[0]
        while size_of_tmp_query_features > 0:

            if size_of_tmp_query_features > 5000:
                avg_prec = calculate_avg_precs(query_features, query_labels, test_features, test_labels, drop_first)

                tmp_query_features = tmp_query_features[5000:, ]
                size_of_tmp_query_features = tmp_query_features.shape[0]
            else:
                avg_prec = calculate_avg_precs(query_features, query_labels, test_features, test_labels, drop_first)
                tmp_query_features = tmp_query_features[tmp_query_features.shape[0]:, ]
                size_of_tmp_query_features = 0

            avg_precs = np.vstack((avg_precs, avg_prec))

    mean_ap = np.mean(avg_precs)
    print('<<<<<<<<< mean_ap')
    print(mean_ap)
    return mean_ap, avg_precs


def calculate_avg_precs(query_features, query_labels, test_features, test_labels, drop_first):
    # split feature vector which has 5000+ dimensions
    if test_features.shape[0] > 5000:
        # initialize dist_mat: [size_of_query_features, 1]
        dist_mat = np.zeros((query_features.shape[0], 1))
        tmp_test_features = test_features
        size_of_tmp_test_features = tmp_test_features.shape[0]
        while size_of_tmp_test_features > 0:

            if size_of_tmp_test_features > 5000:
                tmp_dist_mat = cdist(XA=query_features, XB=tmp_test_features[:5000, ], metric='cosine')
                tmp_test_features = tmp_test_features[5000:, ]
                size_of_tmp_test_features = tmp_test_features.shape[0]
            else:
                tmp_dist_mat = cdist(XA=query_features, XB=tmp_test_features, metric='cosine')
                tmp_test_features = tmp_test_features[tmp_test_features.shape[0]:, ]
                size_of_tmp_test_features = 0

            # dis_mat: [size_of_query_features, m] + tmp_dis_mat:[size_of_query_features, n]
            # -> dist_mat: [size_of_query_features, m+n]
            # dist_mat = np.concatenate([dist_mat, tmp_dist_mat], axis=1)
            dist_mat = np.hstack((dist_mat, tmp_dist_mat))

        # remove first column after all dist_mat's were concatenated
        dist_mat = np.delete(dist_mat, 0, axis=1)
        print("<<<<<<<<<<<< dist_mat after hstack")
        print(type(dist_mat))
        print(dist_mat.shape)
        print(dist_mat)
        print(">>>>>>><>>>>")
    else:
        dist_mat = cdist(XA=query_features, XB=test_features, metric='cosine')

    retrieval_indices = np.argsort(dist_mat, axis=1)
    print("<<<<<<<<<<<< dist_mat")
    print(type(dist_mat))
    print(dist_mat.shape)
    print(dist_mat)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< retrieval_indices")
    print(type(retrieval_indices))
    print(retrieval_indices.shape)
    print(retrieval_indices)
    print(">>>>>>><>>>>")

    # create the retrieval matrix
    # test_labels.shape = [m, 1]
    retr_mat = np.tile(test_labels, (len(query_labels), 1))

    print("<<<<<<<<<<<< test_labels")
    print(type(test_labels))
    print(test_labels.shape)
    print(test_labels)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< retr_mat")
    print(type(retr_mat))
    print(retr_mat.shape)
    print(retr_mat)
    print(">>>>>>><>>>>")

    row_selector = np.transpose(np.tile(np.arange(len(query_labels)), (len(test_labels), 1)))
    print("<<<<<<<<<<<< row_selector")
    print(type(row_selector))
    print(row_selector.shape)
    print(row_selector)
    print(">>>>>>><>>>>")

    retr_mat = retr_mat[row_selector, retrieval_indices]
    print("<<<<<<<<<<<< retr_mat")
    print(type(retr_mat))
    print(retr_mat.shape)
    print(retr_mat)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< qry_labels")
    print(type(query_labels))
    print(query_labels.shape)
    print(query_labels)
    print(">>>>>>><>>>>")

    print("<<<<<<<<<<<< qry_labels.T")
    print(type(np.atleast_2d(query_labels).T))
    print(np.atleast_2d(query_labels).T.shape)
    print(np.atleast_2d(query_labels).T)
    print(">>>>>>><>>>>")

    # create the relevance matrix
    relevance_matrix = retr_mat == np.atleast_2d(query_labels).T
    if drop_first:
        relevance_matrix = relevance_matrix[:, 1:]

    # calculate mAP and APs
    avg_precs = np.array([average_precision(row) for row in relevance_matrix], ndmin=2).flatten()
    print('<<<<<<<<< avg_precs')
    print(avg_precs)

    return avg_precs
