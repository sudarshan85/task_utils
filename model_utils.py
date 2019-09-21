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

def run_iters(data_df, clf_model, params, vectorizer, threshold, workdir, prefix, split_pct=0.15, n_iters=100, start_seed=127):
  targs, preds, probs = [], [], []
  seeds = list(range(start_seed, start_seed + n_iters))
  for seed in tqdm(seeds, desc='Run #'):
    df = set_group_splits(data_df.copy(), group_col='encounter_id', seed=seed)
    train_df = df.loc[df['split'] == 'train', ['note', 'imminent_adm_label']]
    test_df = df.loc[df['split'] == 'test', ['note', 'imminent_adm_label']]
    
    x_train = vectorizer.fit_transform(train_df['note'])
    x_test = vectorizer.transform(test_df['note'])    
    y_train = train_df['imminent_adm_label'].to_numpy()
    y_test = test_df['imminent_adm_label'].to_numpy()
    targs.append(y_test)
    
    if params:      
      clf = clf_model(**params)
    else:
      clf = clf_model

    clf.fit(x_train, y_train)
    pickle.dump(clf, open((workdir/f'models/{prefix}_seed_{seed}.pkl'), 'wb'))
    
    prob = clf.predict_proba(x_test)
    probs.append(prob)
    
    y_pred = (prob[:, 1] > threshold).astype(np.int64)
    preds.append(y_pred)
    
  with open(workdir/f'{prefix}_preds.pkl', 'wb') as f:
    pickle.dump(targs, f)
    pickle.dump(probs, f)
    pickle.dump(preds, f)
