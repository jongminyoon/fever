# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

"""Various retriever utilities."""

import unicodedata
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
from sklearn.utils import murmurhash3_32
from collections import Counter

from nltk.tokenize import RegexpTokenizer, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer




def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)



# ------------------------------------------------------------------------------
# Sparse matrix saving/loading helpers.
# ------------------------------------------------------------------------------


def save_sparse_csr(filename, matrix, metadata=None):
    data = {
        'data': matrix.data,
        'indices': matrix.indices,
        'indptr': matrix.indptr,
        'shape': matrix.shape,
        'metadata': metadata,
    }
    np.savez(filename, **data)


def load_sparse_csr(filename):
    loader = np.load(filename)
    matrix = sp.csr_matrix((loader['data'], loader['indices'],
                            loader['indptr']), shape=loader['shape'])
    return matrix, loader['metadata'].item(0) if 'metadata' in loader else None


# ------------------------------------------------------------------------------
# Token hashing.
# ------------------------------------------------------------------------------


def hash(token, num_buckets):
    """Unsigned 32 bit murmurhash for feature hashing."""
    return murmurhash3_32(token, positive=True) % num_buckets


# ------------------------------------------------------------------------------
# Text cleaning.
# ------------------------------------------------------------------------------

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


STOPWORDS = set(stopwords.words('english'))
ps = PorterStemmer()
reg_tokenizer = RegexpTokenizer(r'[\w-]{3,}')

def get_ngrams(s,n):
    """ make a list of n-gram from a list s, with padding <S> and </S>
    """
    if n > 1 :
        s = ["<S>"] + s + ["</S>"]
    return [" ".join(s[i:i+n]) for i in range(len(s)-n+1)]


def process_sent(sent, stopwords=True, stem=True):
    temp = reg_tokenizer.tokenize(sent)
    if stopwords:
        temp = [w for w in temp if not w in STOPWORDS]
    if stem:
        temp = [ps.stem(w) for w in temp]
    return temp


def process_text(text, stopwords=True, stem=True, ngram=1):
    sent_list = sent_tokenize(text)
    processed_sent_list = [process_sent(sent, stopwords=stopwords, stem=stem) for sent in sent_list]
    final_tokens = []
    for sent in processed_sent_list:
        final_tokens += sent
        for n in range(2,ngram+1):
            final_tokens += get_ngrams(sent, n)

    return final_tokens


def text2mat(text, hash_size, vector=False):
    """ text : list of sentences or a single string
        if vector=False, converts the single string into matrix.
        if vector=True, converts the single string into vector """
    if type(text) is not list:
        if vector:
            text = [text]
        else:
            text = sent_tokenize(text)
    sents = [process_sent(sent) for sent in text]
    counts = [Counter([hash(w, hash_size) for w in sent]) for sent in sents]
    row, col, data = [], [], []
    for i, count in enumerate(counts):
        col.extend(count.keys())
        row.extend([i]*len(count))
        data.extend(count.values())
    vect = sp.csr_matrix(
        (data, (row, col)), shape=(len(counts), hash_size)
    )
    vect = vect.log1p()
    return vect


def closest_sentences(query, text, hash_size, k=5):
    """ text : a list of sentences"""
    scores = text2mat(text, hash_size).dot(text2mat(query, hash_size, vector=True).transpose())
    scores = scores.toarray().squeeze()
    if scores.shape == ():
        return {}
    else:
        inds = pd.Series(scores).nlargest(k).index
        return {i:text[i] for i in inds}

