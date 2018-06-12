# Fact Extraction and VERification

This is an implementation of the [FEVER shared task](http://fever.ai). The goal is to create a system based on a large corpus which determines whether a given claim is supported, refuted, or with not enough information for factual verification. We use pre-processed Wikipedia Pages (June 2017 dump) as the evidence corpus and this is provided by the FEVER task, together with the large training dataset with *****

Our approach can be summarized as follows:



This work can be seen as a simplified implementation and (small) expansion of the pipeline baseline described in the paper: [FEVER: A large-scale dataset for Fact Extraction and VERification.](https://arxiv.org/abs/1803.05355) 

This is a final project for CS224U Natural Language Understanding, Spring 2018 at Stanford University. 


## Installation

Clone the repository

    git clone https://github.com/*******************
    cd fever-baselines

Install requirements (run `export LANG=C.UTF-8` if installation of DrQA fails)

    pip install -r requirements.txt

Download the FEVER dataset from the [website](http://fever.ai/data.html) into the data directory

    mkdir data
    mkdir data/fever-data
    
    # We use the data used in the baseline paper
    wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
    wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
    wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl
    

## Data Preparation
The data preparation consists of three steps: downloading the articles from Wikipedia, indexing these for the Evidence Retrieval and performing the negative sampling for training.  

### 1. Download Wikipedia data

Download the pre-processed Wikipedia articles from [the website](https://sheffieldnlp.github.io/fever/data.html) and unzip it into the data folder.
    
    wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
    unzip wiki-pages.zip -d data
 

### 2. Construct SQLite Database
Construct an SQLite Database. A commercial personal laptop seems not work when dealing with the entire database as a single file so we split the Wikipedia database into a few files too. 
    
    python build_db.py data/wiki-pages data/single --num-files 1
    python build_db.py data/wiki-pages data/fever --num-files 5


### 3. Create Term-Document count matrices and merge
Create a term-document count matrix for each split, and then merge the count matrices.
    
    python build_count_matrix.py data/fever data/index
    python merge_count_matrix.py data/index data/index


### 4. Reweight the count matrix
Two schemes are tried, TF-IDF and PMI.
    
    python reweight_count_matrix.py data/index/count-ngram\=1-hash\=16777216.npz data/index --model tfidf
    python reweight_count_matrix.py data/index/count-ngram\=1-hash\=16777216.npz data/index --model pmi





## Results

The remaining task for FEVER challenge, i.e. document retrieval, sentence selection, sampling for NotEnoughInfo, and RTE training are done in IPython notebook `fever.ipynb` and implementation in `fever.py`. The class `Oracle` reads either TF-IDF or PMI matrix and have methods for finding relevant documents, sentences, etc. given the input claim.


### 1. Document Retrieval
The oracle accuracies for document retrieval for varying number of documents retrieved are

| Accuracy (%) |  Model |      |
|:------------:|:------:|:----:|
|   Num Docs   | TF-IDF |  PMI |
|       1      |  23.2  | 23.2 |
|       3      |  45.5  | 45.5 |
|       5      |  56.9  | 56.9 |
|      10      |  69.0  | 69.0 |


### 2. Sentence Selection

|   Num Docs   | Accuracy (%)|
|:------------:|:-----------:|
|       1      |  51.2       |
|       3      |  67.0       |
|       5      |  72.7       |
|      10      |  81.8       |


### 3. RTE Training
We used logistic classifier with grid cross-validation for best hyperparamters. The details can be found in the final report pdf file in `reports`

#### 1) Word-overlapping feature

|               | Precision | Recall | F1 score |
|:-------------:|:---------:|:------:|:--------:|
| Supported     | 0.337     | 0.798  | 0.455    |
| Refuted       | 0.426     | 0.012  | 0.023    |
| NEI           | 0.362     | 0.326  | 0.343    |
| avg / total   | 0.374     | 0.346  | 0.274    |

#### 2) Word cross-product feature

|               | Precision | Recall | F1 score |
|:-------------:|:---------:|:------:|:--------:|
| Supported     | 0.378     | 0.410  | 0.394    |
| Refuted       | 0.535     | 0.219  | 0.311    |
| NEI           | 0.339     | 0.527  | 0.420    |
| avg / total   | 0.421     | 0.385  | 0.375    |





          
