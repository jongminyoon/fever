# Adapted and modified from https://github.com/sheffieldnlp/fever-baselines/tree/master/src/scripts
# which is adapted from https://github.com/facebookresearch/DrQA/blob/master/scripts/retriever/build_db.py
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.

# Adapted and modified from nli.py, sst.py in https://github.com/cgpotts/cs224u/ 
# by Prof. Christopher Potts for CS224u, Stanford, Spring 2018




import json
import numpy as np
import scipy.sparse as sp
import pandas as pd
import os
import random
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, f1_score
from nltk.tokenize import sent_tokenize
from tqdm import tqdm

import utils
from doc_db import DocDB

DB_PATH = 'data/single/fever.db'
MAT_PATH = 'data/index/tfidf-count-ngram=1-hash=16777216.npz'

class Oracle(object):
    """Oracle from corpus database and term-document matrix.

    Parameters
    ----------
    db_path : str
        Full path to the sqlite3 database file of the corpus.
    mat_path : str
        Full path to the term-document matrix of the corpus.

    """
    def __init__(self, 
            db_path=DB_PATH,
            mat_path=MAT_PATH):
        self.db_path = db_path
        self.mat_path = mat_path
        self.db = DocDB(db_path = self.db_path)
        self.mat, metadata = utils.load_sparse_csr(self.mat_path)

        # doc_freqs, hash_size, ngram, doc_dict
        for k, v in metadata.items():
            setattr(self, k, v)


    def __str__(self):
        return """FEVER Oracle\nDatabase path = {}\nTerm-Document matrix path = {}""".format(
            self.db_path, self.mat_path)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items()}
        return """"FEVER Oracle({})""".format(d)


    def closest_docs(self, query, k=3):
        temp = self.mat.transpose().dot(utils.text2mat(query, self.hash_size, vector=True).transpose())
        inds = pd.Series(temp.toarray().squeeze()).nlargest(k).index
        return [self.doc_dict[1][ind] for ind in inds]


    def doc_ids2texts(self, doc_ids):
        return [self.db.get_doc_text(doc_id) for doc_id in doc_ids]

    def get_sentence(self, doc_id, sent_num):
        temp = sent_tokenize(self.db.get_doc_text(doc_id))
        if len(temp) > sent_num:
            return temp[sent_num]
        else:
            return temp[-1]

    def choose_sents_from_doc_ids(self, query, doc_ids, k=3):
        id_tuple = []
        texts = []
        for doc_id in doc_ids:
            sents = sent_tokenize(self.db.get_doc_text(doc_id))
            for j, sent in enumerate(sents):
                id_tuple.append((doc_id,j))
                texts.append(sent)
        chosen_sents = utils.closest_sentences(query, texts, self.hash_size, k=k)
        return {id_tuple[i]:sent for i, sent in chosen_sents.items()}
            
    def read(self, query, num_sents=3, num_docs=3):
        doc_ids = self.closest_docs(query, k=num_docs)
        return self.choose_sents_from_doc_ids(query, doc_ids, k=num_sents)










class Example(object):
    """For processing examples from FEVER.

    Parameters
    ----------
    d : dict
        Derived from a JSON line in one of the corpus files. Each
        key-value pair becomes an attribute-value pair for the
        class. 

    """
    def __init__(self, d):
        for k, v in d.items():
            setattr(self, k, v)

    def __str__(self):
        return """{}\n{}\n{}""".format(
            self.claim, self.verifiable, self.label)

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"FEVER Example({})""".format(d)

    def get_evidence_ids_for_retrieval_test(self):
        if not hasattr(self, 'label') or self.label == 'NOT ENOUGH INFO':
            return None
        return [(ev[2], ev[3]) for evi in self.evidence for ev in evi]

    def get_evidence_ids(self):
        return [(ev[2], ev[3]) for evi in self.evidence for ev in evi]


class Reader(object):
    """Reader for FEVER data.

    Parameters
    ----------
    src_filename : str
        Full path to the file to process.
    samp_percentage : float or None
        If not None, randomly sample approximately this percentage
        of lines.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    """
    def __init__(self,
            src_filename,
            samp_percentage=None,
            random_state=None):
        self.src_filename = src_filename
        self.samp_percentage = samp_percentage
        self.random_state = random_state

    def read(self):
        """Primary interface.

        Yields
        ------
        `Example`

        """
        random.seed(self.random_state)
        for line in open(self.src_filename):
            if (not self.samp_percentage) or random.random() <= self.samp_percentage:
                d = json.loads(line)
                ex = Example(d)
                yield ex

    def __repr__(self):
        d = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        return """"FEVER Reader({})""".format(d)


FEVER_HOME = os.path.join("data", "fever-data")

class TrainReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "train.jsonl")
        super().__init__(src_filename, **kwargs)


class DevReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "dev.jsonl")
        super().__init__(src_filename, **kwargs)

class TestReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "test.jsonl")
        super().__init__(src_filename, **kwargs)

class SampledTrainReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "train_sampled.jsonl")
        super().__init__(src_filename, **kwargs)


class SampledDevReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "dev_sampled.jsonl")
        super().__init__(src_filename, **kwargs)

class SampledTestReader(Reader):
    def __init__(self, fever_home=FEVER_HOME, **kwargs):
        src_filename = os.path.join(
            fever_home, "test_sampled.jsonl")
        super().__init__(src_filename, **kwargs)


def doc_retrieval_accuracy(reader, oracle, num_docs = 3):
    """Compute document retrieval accurary.

    Parameters
    ----------
    reader : `Reader` instance or one of its subclasses.
    oracle : returns sentences to be used as evidences 
        for verifying the claim

    """

    ex_count = 0
    doc_count = 0
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        ev_ids = ex.get_evidence_ids_for_retrieval_test()
        if ev_ids == None:
            continue
        oracle_ev = oracle.closest_docs(claim, k=num_docs)
        if set(oracle_ev) & set([ev_id[0] for ev_id in ev_ids]):
            doc_count +=1
        ex_count += 1

    print("Num_docs = {}, accuracy {}/{} = {}".format(num_docs, doc_count, ex_count, doc_count/ex_count))
    return doc_count/ex_count


def sentence_selection_accuracy(reader, oracle, num_sents = 3):
    """Compute sentence selection accurary.

    Parameters
    ----------
    reader : `Reader` instance or one of its subclasses.
    oracle : returns sentences to be used as evidences 
        for verifying the claim

    """

    ex_count = 0
    sent_count = 0
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        ev_ids = ex.get_evidence_ids_for_retrieval_test()
        if ev_ids == None:
            continue
        doc_ids = set([ev_id[0] for ev_id in ev_ids])
        id_tuples = oracle.choose_sents_from_doc_ids(claim, doc_ids, k=num_sents).keys()
        if set(id_tuples) & set(ev_ids):
            sent_count +=1
        ex_count += 1

    print("Num_sents = {}, accuracy {}/{} = {}".format(num_sents, sent_count, ex_count, sent_count/ex_count))
    return sent_count/ex_count




def build_dataset_from_scratch(reader, phi, oracle, vectorizer=None, vectorize=True):
    """Create a dataset for training classifiers using `sklearn`.

    Parameters
    ----------
    reader : `Reader` instance or one of its subclasses.
    phi : feature function
        Maps two strings (claim, evidence) to count dictionaries 
        if vectorize=True, or to vectors if vectorize=False
    oracle : returns sentences to be used as evidences 
        for verifying the claim
    assess_reader : `Reader` or one of its subclasses.
        If None, then random train/test splits are performed.
    vectorizer : `sklearn.feature_extraction.DictVectorizer`
        If this is None, then a new `DictVectorizer` is created and
        used to turn the list of dicts created by `phi` into a
        feature matrix. This happens when we are training.
        If this is not None, then it's assumed to be a `DictVectorizer`
        and used to transform the list of dicts. This happens in
        assessment, when we take in new instances and need to
        featurize them as we did in training.
    vectorize : bool
        Whether or not to use a `DictVectorizer` to create the feature
        matrix. If False, then it is assumed that `phi` does this,
        which is appropriate for models that featurize their own data.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the original tree pairs, for error analysis).

    """
    feats = []
    labels = []
    raw_examples = []
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        evidence = oracle.read(claim).values()
        label = ex.label
        d = phi(claim, evidence)
        feats.append(d)
        labels.append(label)
        raw_examples.append((claim, evidence))
    if vectorize:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feats)
        else:
            feat_matrix = vectorizer.transform(feats)
    else:
        # feat_matrix = feats
        feat_matrix = sp.vstack(feats)
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}



def build_dataset(reader, phi, oracle, vectorizer=None, vectorize=True):
    """Create a dataset for training classifiers using `sklearn`.

    Parameters
    ----------
    reader : `Reader` instance or one of its subclasses.
    phi : feature function
        Maps two strings (claim, evidence) to count dictionaries 
        if vectorize=True, or to vectors if vectorize=False
    oracle : returns sentences to be used as evidences 
        for verifying the claim
    assess_reader : `Reader` or one of its subclasses.
        If None, then random train/test splits are performed.
    vectorizer : `sklearn.feature_extraction.DictVectorizer`
        If this is None, then a new `DictVectorizer` is created and
        used to turn the list of dicts created by `phi` into a
        feature matrix. This happens when we are training.
        If this is not None, then it's assumed to be a `DictVectorizer`
        and used to transform the list of dicts. This happens in
        assessment, when we take in new instances and need to
        featurize them as we did in training.
    vectorize : bool
        Whether or not to use a `DictVectorizer` to create the feature
        matrix. If False, then it is assumed that `phi` does this,
        which is appropriate for models that featurize their own data.

    Returns
    -------
    dict
        A dict with keys 'X' (the feature matrix), 'y' (the list of
        labels), 'vectorizer' (the `DictVectorizer`), and
        'raw_examples' (the original tree pairs, for error analysis).

    """
    feats = []
    labels = []
    raw_examples = []
    
    total_len = len(set(reader.read()))
    for ex in tqdm(reader.read(), total=total_len, unit="examples", desc = 'Reading from dataset'):
        claim = ex.claim
        ev_ids = ex.get_evidence_ids()
        sents = [oracle.get_sentence(ev_id[0], ev_id[1]) for ev_id in ev_ids]
        d = phi(claim, sents)
        feats.append(d)
        labels.append(ex.label)
        raw_examples.append((claim, sents))
    if vectorize:
        if vectorizer == None:
            vectorizer = DictVectorizer(sparse=True)
            feat_matrix = vectorizer.fit_transform(feats)
        else:
            feat_matrix = vectorizer.transform(feats)
    else:
        # feat_matrix = feats
        feat_matrix = sp.vstack(feats)
    return {'X': feat_matrix,
            'y': labels,
            'vectorizer': vectorizer,
            'raw_examples': raw_examples}



def safe_macro_f1(y, y_pred):
    """Macro-averaged F1, forcing `sklearn` to report as a multiclass
    problem even when there are just two classes. `y` is the list of
    gold labels and `y_pred` is the list of predicted labels."""
    return f1_score(y, y_pred, average='macro', pos_label=None)

def experiment(
        train_reader,
        phi,
        oracle,
        train_func,
        assess_reader=None,
        train_size=0.7,
        score_func=safe_macro_f1,
        vectorize=True,
        verbose=True,
        random_state=None):
    """Generic experimental framework for FEVER. Either assesses with a
    random train/test split of `train_reader` or with `assess_reader` if
    it is given.

    Parameters
    ----------
    train_reader : `Reader` instance or one of its subclasses.
        Iterator for training data.
    phi : feature function
        Maps two strings (claim, evidence) to count dictionaries 
        if vectorize=True, or to vectors if vectorize=False
    oracle : returns sentences to be used as evidences 
        for verifying the claim
    train_func : model wrapper
        Any function that takes a feature matrix and a label list
        as its values and returns a fitted model with a `predict`
        function that operates on feature matrices.
    assess_reader : None, or `Reader` or one of its subclasses
        If None, then the data from `train_reader` are split into
        a random train/test split, with the the train percentage
        determined by `train_size`.
    train_size : float
        If `assess_reader` is None, then this is the percentage of
        `train_reader` devoted to training. If `assess_reader` is
        not None, then this value is ignored.
    score_metric : function name
        This should be an `sklearn.metrics` scoring function. The
        default is weighted average F1 (macro-averaged F1). 
    vectorize : bool
       Whether to use a DictVectorizer. Set this to False for
       deep learning models that process their own input.
    verbose : bool
        Whether to print out the model assessment to standard output.
        Set to False for statistical testing via repeated runs.
    random_state : int or None
        Optionally set the random seed for consistent sampling.

    Prints
    -------
    To standard output, if `verbose=True`
        Model precision/recall/F1 report. 

    Returns
    -------
    float
        The overall scoring metric as determined by `score_metric`.

    """
    # Train dataset:
    train = build_dataset(
        train_reader,
        phi,
        oracle,
        vectorizer=None,
        vectorize=vectorize)
    # Manage the assessment set-up:
    X_train = train['X']
    y_train = train['y']
    X_assess = None
    y_assess = None
    if assess_reader == None:
         X_train, X_assess, y_train, y_assess = train_test_split(
             X_train, y_train, train_size=train_size, test_size=None,
             random_state=random_state)
    else:
        # Assessment dataset using the training vectorizer:
        assess = build_dataset(
            assess_reader,
            phi,
            oracle,
            vectorizer=train['vectorizer'],
            vectorize=vectorize)
        X_assess, y_assess = assess['X'], assess['y']
    # Train:
    mod = train_func(X_train, y_train)
    # Predictions:
    predictions = mod.predict(X_assess)
    # Report:
    if verbose:
        print(classification_report(y_assess, predictions, digits=3))
    # Return the overall score:
    return score_func(y_assess, predictions)





def fit_classifier_with_crossvalidation(X, y, basemod, cv, param_grid, scoring='f1_macro'):
    """Fit a classifier with hyperparmaters set via cross-validation.

    Parameters
    ----------
    X : 2d np.array
        The matrix of features, one example per row.
    y : list
        The list of labels for rows in `X`.
    basemod : an sklearn model class instance
        This is the basic model-type we'll be optimizing.
    cv : int
        Number of cross-validation folds.
    param_grid : dict
        A dict whose keys name appropriate parameters for `basemod` and
        whose values are lists of values to try.
    scoring : value to optimize for (default: f1_macro)
        Other options include 'accuracy' and 'f1_micro'. See
        http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter

    Prints
    ------
    To standard output:
        The best parameters found.
        The best macro F1 score obtained.

    Returns
    -------
    An instance of the same class as `basemod`.
        A trained model instance, the best model found.

    """
    # Find the best model within param_grid:
    crossvalidator = GridSearchCV(basemod, param_grid, cv=cv, scoring=scoring)
    crossvalidator.fit(X, y)
    # Report some information:
    print("Best params", crossvalidator.best_params_)
    print("Best score: %0.03f" % crossvalidator.best_score_)
    # Return the best model found:
    return crossvalidator.best_estimator_
