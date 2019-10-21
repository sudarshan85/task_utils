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

from tqdm import trange

from .data_utils import set_group_splits
from .tokenizer import tokenize
from sklearn.model_selection import RandomizedSearchCV

def get_best_params(clf, x_train, y_train, param_space, **kwargs):
  param_search = RandomizedSearchCV(clf, param_space, **kwargs)
  param_search.fit(x_train, y_train)
  
  return param_search.best_params_, pd.DataFrame(param_search.cv_results_)

def run_iters(data_df, clf_model, clf_params, vectorizer, threshold, start_seed, split_pct=0.15, n_iters=100):
  clfs, targs, preds, probs = [], [], [], []
  t = trange(start_seed, start_seed + n_iters, desc='Run #', leave=True)
  # seeds = list(range(start_seed, start_seed + n_iters))

  for seed in t:
    t.set_description(f"Run # (seed {seed})")
    df = set_group_splits(data_df.copy(), group_col='hadm_id', seed=seed, pct=split_pct)
    train_df = df.loc[df['split'] == 'train', ['note', 'imi_adm_label']]
    test_df = df.loc[df['split'] == 'test', ['note', 'imi_adm_label']]

    x_train = vectorizer.fit_transform(train_df['note'])
    x_test = vectorizer.transform(test_df['note'])    
    y_train = train_df['imi_adm_label'].to_numpy()
    y_test = test_df['imi_adm_label'].to_numpy()
    targs.append(y_test)

    clf = clf_model(**clf_params)
    clf.fit(x_train, y_train)
    clfs.append(clf)

    prob = clf.predict_proba(x_test)
    probs.append(prob)

    y_pred = (prob[:, 1] > threshold).astype(np.int64)
    preds.append(y_pred)
  
  return clfs, targs, probs, preds
