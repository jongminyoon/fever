# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import importlib.util

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
import utils

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Build db")
# TODO add time for logging

# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------

def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document
            documents.append((utils.normalize(doc['id']), doc['text']))
    return documents


def store_contents(data_path, save_path, num_workers=4, num_files = 5):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
        num_workers: Number of parallel processes to use when reading docs.
        num_files: Split db in to num_files files.
    """

    logger.info('Reading into database...')

    files = [f for f in utils.iter_files(data_path)]

    if num_files == 1:
        filelist = [files]
    else:
        one_length = len(files) // num_files + 1

        filelist = [[files[i*one_length+j] for j in range(one_length)] for i in range(num_files-1)]
        filelist.append(files[one_length*(num_files-1):])

    for i, files in enumerate(filelist):
        logger.info('Building %i-th db...' % i)

        temp_save_path = os.path.join(save_path, 'fever%i.db' % i)

        if os.path.isfile(temp_save_path):
            raise RuntimeError('%s already exists! Not overwriting.' % temp_save_path)

        conn = sqlite3.connect(temp_save_path)
        c = conn.cursor()
        c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

        workers = ProcessPool(num_workers)
        count = 0
        with tqdm(total=len(files)) as pbar:
            for pairs in tqdm(workers.imap_unordered(get_contents, files)):
                count += len(pairs)
                c.executemany("INSERT INTO documents VALUES (?,?)", pairs)
                pbar.update()
        logger.info('Read %d docs.' % count)
        logger.info('Committing...')
        conn.commit()
        conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='path/to/data')
    parser.add_argument('save_path', type=str, help='path/to/saved')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of CPU processes (for tokenizing, etc)')
    parser.add_argument('--num-files', type=int, default=None,
                        help='Number of db files')
    args = parser.parse_args()

    save_dir = args.save_path
    if not os.path.exists(save_dir):
        logger.info("Save directory doesn't exist. Making {0}".format(save_dir))
        os.makedirs(save_dir)

    store_contents(
        args.data_path, args.save_path, args.num_workers, args.num_files
    )


# python build_db.py data/wiki-pages data/fever