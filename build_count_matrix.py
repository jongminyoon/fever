# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

"""A script to build term-document count matrices for retrieval."""


import numpy as np
import scipy.sparse as sp
import argparse
import os
import math
import logging


from multiprocessing import Pool as ProcessPool
from multiprocessing.util import Finalize
from functools import partial
from collections import Counter

import utils
from doc_db import DocDB


# ------------------------------------------------------------------------------
# Multiprocessing functions
# ------------------------------------------------------------------------------

DOC2IDX = None
PROCESS_DB = None


def init(db_class, db_opts):
    global PROCESS_DB
    PROCESS_DB = db_class(**db_opts)
    Finalize(PROCESS_DB, PROCESS_DB.close, exitpriority=100)


def fetch_text(doc_id):
    global PROCESS_DB
    return PROCESS_DB.get_doc_text(doc_id)

# ------------------------------------------------------------------------------
# Build article --> word count sparse matrix.
# ------------------------------------------------------------------------------


def count(ngram, hash_size, doc_id):
    """Fetch the text of a document and compute hashed ngrams counts."""
    global DOC2IDX
    row, col, data = [], [], []

    # Get ngrams after tokenizing and processing (i.e. stopword/punctuation filtering)
    ngrams = utils.process_text(utils.normalize(fetch_text(doc_id)), 
                                stopwords=True, stem=True, ngram=ngram)

    # Hash ngrams and count occurences
    counts = Counter([utils.hash(gram, hash_size) for gram in ngrams])

    # Return in sparse matrix data format.
    row.extend(counts.keys())
    col.extend([DOC2IDX[doc_id]] * len(counts))
    data.extend(counts.values())
    return row, col, data


def get_count_matrix(args, db, db_opts):
    """Form a sparse word to document count matrix (inverted index).

    M[i, j] = # times word i appears in document j.
    """
    # Map doc_ids to indexes
    global DOC2IDX
    db_class = DocDB
    with db_class(**db_opts) as doc_db:
        doc_ids = doc_db.get_doc_ids()
    DOC2IDX = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    # Setup worker pool
    workers = ProcessPool(
        args.num_workers,
        initializer=init,
        initargs=(db_class, db_opts)
    )

    # Compute the count matrix in steps (to keep in memory)
    logger.info('Mapping...')
    row, col, data = [], [], []
    step = max(int(len(doc_ids) / 19), 1)
    batches = [doc_ids[i:i + step] for i in range(0, len(doc_ids), step)]
    _count = partial(count, args.ngram, args.hash_size)
    for i, batch in enumerate(batches):
        logger.info('-' * 25 + 'Batch %d/%d' % (i + 1, len(batches)) + '-' * 25)
        for b_row, b_col, b_data in workers.imap_unordered(_count, batch):
            row.extend(b_row)
            col.extend(b_col)
            data.extend(b_data)
    workers.close()
    workers.join()

    logger.info('Creating sparse matrix...')
    count_matrix = sp.csr_matrix(
        (data, (row, col)), shape=(args.hash_size, len(doc_ids))
    )
    count_matrix.sum_duplicates()
    return count_matrix, (DOC2IDX, doc_ids)


# ------------------------------------------------------------------------------
# Transform count matrix to different forms.
# ------------------------------------------------------------------------------




def get_doc_freqs(cnts):
    """Return word --> # of docs it appears in."""
    binary = (cnts > 0).astype(int)
    freqs = np.array(binary.sum(1)).squeeze()
    return freqs



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Build count matrix")


    parser = argparse.ArgumentParser()
    parser.add_argument('db_path', type=str, default=None,
                        help='Path to sqlite db holding document texts')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    parser.add_argument('--ngram', type=int, default=1,
                        help=('Use up to N-size n-grams '
                              '(e.g. 2 = unigrams + bigrams)'))
    parser.add_argument('--hash-size', type=int, default=int(math.pow(2, 24)),
                        help='Number of buckets to use for hashing ngrams')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()


    db_files = [f for f in utils.iter_files(args.db_path)]

    for i, f in enumerate(db_files):
        logger.info('Processing file %i...' % i)
        
        logger.info('Counting words...')

        count_matrix, doc_dict = get_count_matrix(
            args, 'sqlite', {'db_path': f}
        )

        logger.info('Getting word-doc frequencies...')
        freqs = get_doc_freqs(count_matrix)

        basename = os.path.splitext(os.path.basename(f))[0]
        basename += ('-ngram=%d-hash=%d' %
                     (args.ngram, args.hash_size))

        if not os.path.exists(args.out_dir):
            logger.info("Creating data directory")
            os.makedirs(args.out_dir)

        filename = os.path.join(args.out_dir, basename)

        logger.info('Saving to %s.npz' % filename)
        metadata = {
            'doc_freqs': freqs,
            'hash_size': args.hash_size,
            'ngram': args.ngram,
            'doc_dict': doc_dict
        }

        utils.save_sparse_csr(filename, count_matrix, metadata)