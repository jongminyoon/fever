# Merge count matrices into a single file.

import numpy as np
import scipy.sparse as sp
import argparse
import os
import logging

from collections import Counter

import utils



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("Merge count matrix")


    parser = argparse.ArgumentParser()
    parser.add_argument('ct_path', type=str, default=None,
                        help='Path to count matrices')
    parser.add_argument('out_dir', type=str, default=None,
                        help='Directory for saving output files')
    args = parser.parse_args()

    ct_files = [f for f in utils.iter_files(args.ct_path)]


    logger.info('Loading the zeroth count matrix...')
    mat, metadata = utils.load_sparse_csr(ct_files[0])

    DOC2IDX, doc_ids = metadata['doc_dict']


    for i in range(1, len(ct_files)):

        logger.info('Loading %ith count matrix...' % i)
        nxt_mat, nxt_metadata = utils.load_sparse_csr(ct_files[i])

        if metadata['hash_size'] != nxt_metadata['hash_size']:
            raise RuntimeError('hash_size not equal in %ith file' % i)
        if metadata['ngram'] != nxt_metadata['ngram']:
            raise RuntimeError('ngram not equal in %ith file' % i)

        logger.info('Merging...')
        mat = sp.hstack([mat, nxt_mat])
        logger.info('Finished merging')

        metadata['doc_freqs'] += nxt_metadata['doc_freqs']

        nxt_DOC2IDX, nxt_doc_ids = nxt_metadata['doc_dict']

        if set(doc_ids).intersection(nxt_doc_ids):
            raise RuntimeError('overlapping doc id n %ith file' % i)

        for k in nxt_DOC2IDX.keys():
            nxt_DOC2IDX[k] += len(DOC2IDX)

        DOC2IDX = {**DOC2IDX, **nxt_DOC2IDX}
        doc_ids += nxt_doc_ids


    metadata['doc_dict'] = (DOC2IDX, doc_ids)

    basename = 'count' + ('-ngram=%d-hash=%d' %
                     (metadata['ngram'], metadata['hash_size']))

    if not os.path.exists(args.out_dir):
        logger.info("Creating data directory")
        os.makedirs(args.out_dir)

    filename = os.path.join(args.out_dir, basename)
    
    logger.info('Saving to %s.npz' % filename)
    # sp.save_npz(filename, mat)
    # np.savez(filename+'meta', **metadata)
    mat = mat.tocsr()
    utils.save_sparse_csr(filename, mat, metadata)
