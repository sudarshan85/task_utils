#!/usr/bin/env python

import datetime
import logging
import sys

import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer

from tqdm import tqdm

from .data_utils import set_group_splits
from .tokenizer import tokenize
from sklearn.model_selection import RandomizedSearchCV

def get_best_params(clf, x_train, y_train, param_space, **kwargs):
  param_search = RandomizedSearchCV(clf, param_space, **kwargs)
  param_search.fit(x_train, y_train)
  
  return param_search.best_params_, pd.DataFrame(param_search.cv_results_)

def run_iters(data_df, group_id, clf_model, tokenizer, params, threshold, model_dir, n_iters=100, start_seed=127):
  targs, preds, probs = [], [], []
  tfidf_params = {
  'ngram_range': (1, 2),
  'tokenizer': tokenizer,
  'min_df': 6,
  'max_features': 52_000,
  'binary': True,
  'sublinear_tf': True,  
  }

  seeds = list(range(start_seed, start_seed + n_iters))
  for seed in tqdm(seeds, desc='Run #'):
    df = set_group_splits(data_df, group_col=group_id, seed=seed)
    vectorizer = TfidfVectorizer(**tfidf_params)
