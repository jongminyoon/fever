# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.



"""A script to build the tf-idf and pmi document matrices for retrieval."""

import numpy as np
import scipy.sparse as sp
import argparse
import os
from collections import Counter
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Reweight count matrix")


import utils


def get_freqs(cnts, axis=1):
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(axis)).squeeze()
    return freqs

def get_tfidf_matrix(cnts):
    """Convert the word count matrix into tfidf one.

    tfidf = log(tf + 1) * log((N - freqs + 0.5) / (freqs + 0.5))
    """
    freqs = get_freqs(cnts)
    doccount = cnts.shape[1]
    idfs = np.log((doccount - freqs + 0.5) / (freqs + 0.5))
    idfs[idfs < 0] = 0
    idfs = sp.diags(idfs, 0)
    tfs = cnts.log1p()
    tfidfs = idfs.dot(tfs)
    return tfidfs

def get_pmi_matrix(cnts):
    """Convert the word count matrix into pmi one.

    pmi = log(tf + 1) * total / col_sum / row_sum
    """
    row = np.array(cnts.sum(axis=1)).squeeze()
    col = np.array(cnts.sum(axis=0)).squeeze()
    total = col.sum()
    with np.errstate(divide='ignore'):
        row = total/row
        row[np.isinf(row)] = 0.0
        col = 1.0/col
        col[np.isinf(col)] = 0.0
    row = sp.diags(row, 0)
    col = sp.diags(col, 0)
    
    temp = row.dot(cnts.dot(col))
    temp = temp.log1p()
    return temp



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ct_path', type=str, default=None,
                        help='Path to count matrix file')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--model', type=str, default='tfidf',
                        help=('tfidf or pmi'))

    args = parser.parse_args()

    logger.info('Loading count matrix...')

    count_matrix, metadata = utils.load_sparse_csr(args.ct_path)

    logger.info('Making %s vectors...' % args.model)
    
    if args.model == 'tfidf':
        mat = get_tfidf_matrix(count_matrix)
    elif args.model == 'pmi':
        mat = get_pmi_matrix(count_matrix)
    else:
        raise RuntimeError('Model %s is invalid' % args.model)

    basename = os.path.splitext(os.path.basename(args.ct_path))[0]
    basename = ('%s-' % args.model) + basename

    if not os.path.exists(args.out_dir):
        logger.info("Creating data directory")
        os.makedirs(args.out_dir)

    filename = os.path.join(args.out_dir, basename)

    logger.info('Saving to %s.npz' % filename)
    utils.save_sparse_csr(filename, mat, metadata)
