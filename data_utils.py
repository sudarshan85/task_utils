#!/usr/bin/env python

import pandas as pd
import numpy as np
# from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split, GroupShuffleSplit

# def balanced_sample(x, y, pct=1.0, seed=None):
#   sampler = RandomOverSampler(random_state=seed)
#   x_resample, y_resample = sampler.fit_resample(x, y)
#   x_resample = x_resample.reshape(-1)
#   assert(x_resample.shape[0] == y_resample.shape[0])
#   idxs = np.random.choice(x_resample.shape[0], np.int64(x_resample.shape[0] * pct))
#   return x_resample[idxs], y_resample[idxs]

def test(a,b,cmp,cname=None):
  if cname is None: cname=cmp.__name__
  assert cmp(a,b),f"{cname}:\n{a}\n{b}"

def near(a,b): return np.allclose(a, b, rtol=1e-1, atol=1e-1)
def test_near(a,b): test(a,b,near)

def set_all_splits(df, val_pct, test_pct=0.0, seed=None):
  new_test_pct = np.around(test_pct / (val_pct + test_pct), 2)
  train_pct = 1 - (val_pct + test_pct)
  train_idxs, inter = train_test_split(np.arange(len(df)), test_size=(val_pct + test_pct), random_state=seed)
  val_idxs, test_idxs = train_test_split(inter, test_size=new_test_pct, random_state=seed)

  df['split'] = None
  df.iloc[train_idxs, df.columns.get_loc('split')] = 'train'
  df.iloc[val_idxs, df.columns.get_loc('split')] = 'val'
  df.iloc[test_idxs, df.columns.get_loc('split')] = 'test'

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), train_pct)
  test_near(round(len(df[df['split'] == 'val'])/len(df), 2), val_pct)
  test_near(round(len(df[df['split'] == 'test'])/len(df), 2), test_pct)

  return df

def set_splits_with_sample(df, val_pct, test_pct=0.0, sample_pct=0.0, seed=None):
  new_test_pct = np.around(test_pct / (val_pct + test_pct), 2)
  train_pct = 1 - (val_pct + test_pct)
  train_idxs, inter = train_test_split(np.arange(len(df)), test_size=(val_pct + test_pct), random_state=seed)
  val_idxs, test_idxs = train_test_split(inter, test_size=new_test_pct, random_state=seed)

  df['split'] = None
  df.iloc[train_idxs, df.columns.get_loc('split')] = 'train'
  df.iloc[val_idxs, df.columns.get_loc('split')] = 'val'
  df.iloc[test_idxs, df.columns.get_loc('split')] = 'test'

  if sample_pct > 0.0:
    df['is_sample'] = False
    _, sample_idxs = train_test_split(train_idxs, test_size=sample_pct)
    df.iloc[sample_idxs, df.columns.get_loc('is_sample')] = True
    test_near(round(len(df[df['is_sample']])/len(df), 2), round(sample_pct * train_pct, 2))

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), train_pct)
  test_near(round(len(df[df['split'] == 'val'])/len(df), 2), val_pct)
  test_near(round(len(df[df['split'] == 'test'])/len(df), 2), test_pct)

  return df

def set_two_splits(df, name, pct=0.15, seed=None):
  df['split'] = 'train'
  _, val_idxs = train_test_split(np.arange(len(df)), test_size=pct, random_state=seed)
  df.loc[val_idxs, 'split'] = name

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), 1-pct)
  test_near(round(len(df[df['split'] == name])/len(df), 2), pct)

  return df

def set_bool_split(df, pct=0.15, seed=None):
  df['is_valid'] = False
  _, val_idxs = train_test_split(np.arange(len(df)), test_size=pct, random_state=seed)
  df.loc[val_idxs, 'is_valid'] = True

  test_near(round(len(df[df['is_valid'] == False])/len(df), 2), 1-pct)
  test_near(round(len(df[df['is_valid'] == True])/len(df), 2), pct)

  return df

def set_group_splits(df, group_col, pct=0.15, name='test', seed=None):
  df['split'] = 'train'
  train_idxs, test_idxs = next(GroupShuffleSplit(test_size=pct, n_splits=2, random_state=seed).split(df, groups=df[group_col]))
  df.loc[test_idxs, 'split'] = name

  assert(set(df.loc[(df['split'] == 'train')][group_col].unique().tolist()).intersection(df[(df['split'] == name)][group_col].unique().tolist()) == set())
  test_near(len(df.loc[(df['split'] == 'train')])/len(df), 1-pct)
  test_near(len(df.loc[(df['split'] == name)])/len(df), pct)

  return df  

def set_group_all_splits(df, group_col, val_pct, test_pct, seed=None):
  new_test_pct = np.around(test_pct / (val_pct + test_pct), 2)
  train_pct = 1 - (val_pct + test_pct)
  split_df = set_group_splits(df, group_col, (val_pct + test_pct), seed=seed)
  train_df = split_df.loc[(split_df['split'] == 'train')].reset_index()
  inter_df = split_df.loc[(split_df['split'] == 'test')].reset_index()
  split_df = set_group_splits(inter_df, group_col, new_test_pct, seed=seed)
  val_df = split_df.loc[(split_df['split'] == 'train')].reset_index()
  val_df['split'] = 'val'
  test_df = split_df.loc[(split_df['split'] == 'test')].reset_index()
  
  df = pd.concat([train_df, val_df, test_df], axis=0, sort=False)
  
  assert(set(df.loc[(df['split'] == 'train')][group_col].unique().tolist()).intersection(df[(df['split'] == 'val')][group_col].unique().tolist()) == set())
  assert(set(df.loc[(df['split'] == 'train')][group_col].unique().tolist()).intersection(df[(df['split'] == 'test')][group_col].unique().tolist()) == set())
  assert(set(df.loc[(df['split'] == 'val')][group_col].unique().tolist()).intersection(df[(df['split'] == 'test')][group_col].unique().tolist()) == set())

  test_near(round(len(df[df['split'] == 'train'])/len(df), 2), 1-(val_pct+test_pct))
  test_near(round(len(df[df['split'] == 'val'])/len(df), 2), val_pct)
  test_near(round(len(df[df['split'] == 'test'])/len(df), 2), test_pct)

  return df