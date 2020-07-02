'''
by chen ma
https://github.com/allenjack/HGN
'''

from interactions import Interactions
from eval_metrics import *

import argparse
import logging
from time import time
import datetime
import torch
import pdb
import pickle

def evaluation(ham, train, test_set, config, device, topk=20):
    num_users = train.num_users
    num_items = train.num_items
    batch_size = 64
    num_batches = int(num_users / batch_size) + 1
    user_indexes = np.arange(num_users)
    item_indexes = np.arange(num_items)
    pred_list = None
    train_matrix = train.tocsr()
    test_sequences = train.test_sequences.sequences

    for batchID in range(num_batches):
        start = batchID * batch_size
        end = start + batch_size

        if batchID == num_batches - 1:
            if start < num_users:
                end = num_users
            else:
                break

        batch_user_index = user_indexes[start:end]

        batch_test_sequences = test_sequences[batch_user_index]
        batch_test_sequences = np.atleast_2d(batch_test_sequences)

        batch_test_sequences = torch.from_numpy(batch_test_sequences).type(torch.LongTensor).to(device)

        item_ids = torch.from_numpy(item_indexes).type(torch.LongTensor).to(device)
        batch_user_ids = torch.from_numpy(np.array(batch_user_index)).type(torch.LongTensor).to(device)

        rating_pred = ham(batch_test_sequences, batch_user_ids, item_ids, True)
        rating_pred = rating_pred.cpu().data.numpy().copy()
        rating_pred[train_matrix[batch_user_index].toarray() > 0] = 0

        # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
        ind = np.argpartition(rating_pred, -topk)
        ind = ind[:, -topk:]
        arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
        arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
        batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

        if batchID == 0:
            pred_list = batch_pred_list
        else:
            pred_list = np.append(pred_list, batch_pred_list, axis=0)

    precision, recall, MAP, ndcg = [], [], [], []

    for k in [5, 10, 15, 20]:
        precision.append(precision_at_k(test_set, pred_list, k))
        recall.append(recall_at_k(test_set, pred_list, k))
        MAP.append(mapk(test_set, pred_list, k))
        ndcg.append(ndcg_k(test_set, pred_list, k))

    return precision, recall, MAP, ndcg

def negsamp_vectorized_bsearch_preverif(pos_inds, n_items, n_samp=32):
    """ Pre-verified with binary search
    `pos_inds` is assumed to be ordered
    reference: https://tech.hbc.com/2018-03-23-negative-sampling-in-numpy.html
    """
    raw_samp = np.random.randint(0, n_items - len(pos_inds), size=n_samp)
    pos_inds_adj = pos_inds - np.arange(len(pos_inds))
    neg_inds = raw_samp + np.searchsorted(pos_inds_adj, raw_samp, side='right')
    return neg_inds


def generate_negative_samples(train_matrix, num_neg=3, num_sets=10):
    neg_samples = []
    for user_id, row in enumerate(train_matrix):
        pos_ind = row.indices
        neg_sample = negsamp_vectorized_bsearch_preverif(pos_ind, train_matrix.shape[1], num_neg * num_sets)
        neg_samples.append(neg_sample)

    return np.asarray(neg_samples).reshape(num_sets, train_matrix.shape[0], num_neg)

