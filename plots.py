#!/usr/bin/env python

import warnings
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from matplotlib import ticker
import seaborn as sns

from typing import List
from scipy import interp
from wordcloud import WordCloud
from spacy.lang.en.stop_words import STOP_WORDS
from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix
from sklearn.metrics import average_precision_score, precision_recall_curve

def plot_model_roc(ax, y_true, prob):
  fpr, tpr, _ = roc_curve(y_true, prob)
  ax.set_ylabel('Sensitivity')
  ax.set_xlabel('1 - Specificity')
  ax.plot([0, 1], [0, 1], linestyle='--')
  ax.plot(fpr, tpr, marker='.')
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)

def get_wordcloud(feature_names, scores, n_words='all'):
  if n_words == 'all':
    n_words = len(feature_names)

  p = re.compile('^[a-z\s]+$')
  neg_dict, pos_dict = {}, {}
  for word, score in zip(feature_names, scores):
    word = word.lower()
    if len(word) > 7 and word not in STOP_WORDS:
      if p.match(word):      
        neg_dict[word] = 1 - score
        pos_dict[word] = score

  neg_cloud = WordCloud(width=400, height=400, background_color='white', max_words=n_words, max_font_size=40, relative_scaling=0.5).generate_from_frequencies(neg_dict)
  pos_cloud = WordCloud(width=400, height=400, background_color='white', max_words=n_words, max_font_size=60, relative_scaling=0.5).generate_from_frequencies(pos_dict)  
    
  return neg_cloud, pos_cloud

def print_top_words(feature_names: List[str], probs: np.ndarray, N: int):
  words = sorted(zip(probs, feature_names), reverse=True)
  pos = words[:N]
  neg = words[:-(N + 1):-1]

  print("Words associated with imminent threat: ")
  for feat in pos:
    print(np.round(feat[0], 2), feat[1])

  print("***********************************************")
  print("Words associated with not imminent threat: ")
  for feat in neg:
    print(np.round(feat[0], 2), feat[1])

def plot_prob(ax, df, threshold, starting_day, ending_day, interval_hours, is_agg=False, is_log=False):
  if starting_day > 0:
    warnings.warn(f"starting_day ({starting_day}) must be negative. Converting it to negative")
    starting_day = -starting_day

  if ending_day > 0:
    warnings.warn(f"ending_day ({ending_day}) must be negative. Converting it to negative")
    ending_day = -ending_day

  if ending_day < starting_day:
    warnings.warn(f"starting_day ({starting_day}) must be less than ending_day ({ending_day}). Swapping values.")
    starting_day, ending_day = ending_day, starting_day

  high = pd.to_timedelta(ending_day, unit='d')
  low = pd.to_timedelta(starting_day, unit='d')  
  plot_data = df.loc[(df['relative_charttime'] > low) & (df['relative_charttime'] < high)][['relative_charttime', 'prob']].copy()
  plot_data['interval'] = ((plot_data['relative_charttime'].apply(lambda curr_time: int((curr_time - df['relative_charttime'].max())/pd.to_timedelta(interval_hours, unit='h')))))/2

  if is_agg:
    plot_data = plot_data[['interval', 'prob']].groupby(['interval']).agg(lambda x: np.average(x, weights=plot_data.loc[x.index, 'prob']))

  plot_data.reset_index(inplace=True)
  if is_log:
    plot_data['interval'] = -np.log1p(-plot_data['interval'])

  ax.axhline(y=threshold, label=f'Threshold = {threshold}', linestyle='--', color='r')
  sns.lineplot(x='interval', y='prob', data=plot_data, ax=ax)
  # ax.set_xlabel(f'Time to ICU (days)')
  # ax.set_ylabel('Probability')
  ax.set_xlabel('')
  ax.set_ylabel('')
  ax.legend(loc='upper left')
  ax.xaxis.set_major_locator(ticker.MultipleLocator(1))

def plot_mean_roc(ax, y_true, y_probas):
  y_true = np.array(y_true)
  y_probas = np.array(y_probas)
  probas = y_probas
  
  tprs = []
  base_fpr = np.linspace(0, 1, 100)
  rocs = []

  for i, (y_test, prob) in enumerate(zip(y_true, y_probas)):
    fpr, tpr, _ = roc_curve(y_test, prob[:, 1])
    rocs.append(auc(fpr, tpr))
    tpr = interp(base_fpr, fpr, tpr)    
    tpr[0] = 0.0
    tprs.append(tpr)

  tprs = np.array(tprs)
  mean_tprs = tprs.mean(axis=0)
  std = tprs.std(axis=0)

  tprs_upper = np.minimum(mean_tprs + std, 1)
  tprs_lower = mean_tprs - std

  mean_roc = np.mean(np.array(rocs), axis=0)
  color = plt.cm.get_cmap('nipy_spectral')(0.5)
  ax.plot(base_fpr, mean_tprs, lw=2, color=color, label=f'Average AUC = {mean_roc:0.2f}')

  ax.plot([0, 1], [0, 1], 'k--', lw=2)
  ax.set_xlim([0.0, 1.0])
  ax.set_ylim([0.0, 1.05])
  ax.set_xlabel('False Positive Rate', fontsize='medium')
  ax.set_ylabel('True Positive Rate', fontsize='medium')
  ax.tick_params(labelsize='medium')
  ax.legend(loc='lower right', fontsize='medium')
  
  return mean_roc  

def plot_thresh_range(ax, y_true, prob, lower=0, upper=1, n_vals=5):
  metrics = np.zeros((4, n_vals))
  thresh_range = np.round(np.linspace(lower, upper, n_vals), 2)

  for i, thresh in enumerate(thresh_range):
    y_pred = (prob > thresh).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred)
    tn,fp,fn,tp = cm[0][0],cm[0][1],cm[1][0],cm[1][1]
    metrics[0][i] = np.round(tp/(tp+fn), 3)
    metrics[1][i] = np.round(tn/(tn+fp), 3)
    metrics[2][i] = np.round(tp/(tp+fp), 3)
    metrics[3][i] = np.round(tn/(tn+fn), 3)

  df = pd.DataFrame(metrics, index=['sensitivity', 'specificity', 'ppv', 'npv'], columns=thresh_range)
  df=df.stack().reset_index()
  df.columns = ['Metric','Threshold','Value']
  ax = sns.pointplot(x='Threshold', y='Value', hue='Metric',data=df)
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5)
  ax.legend(loc='upper right')

def threshold_guide(y_test, prob, ax, metric='youden', beta=None, n_vals=10, granularity=10):
  thresh_range = np.round(np.linspace(0, 1, n_vals), 2)
  cms = np.zeros((n_vals, 2, 2))
  
  for i, thresh in enumerate(thresh_range):
    y_pred = (prob > thresh).astype(np.int64)
    cms[i] = confusion_matrix(y_test, y_pred)

  se = cms[:, 1, 1] / (cms[:, 1, 1] + cms[:, 1, 0])
  sp = cms[:, 0, 0] / (cms[:, 0, 0] + cms[:, 0, 1])
  ppv = cms[:, 1, 1] / (cms[:, 1, 1] + cms[:, 0, 1])
    
  if metric == 'youden':
    metrics = se + sp
    metric = 'Youden Index'
  elif metric == 'weighted_youden':
    if beta is None:
      raise NameError(f"Weight value beta not specified for {metric}")
    metrics = beta * se + (1 - beta) * sp
    metric = f'Youden Index with weight {beta}'
  elif metric == 'f1':     
    metrics = (2 * se * ppv) / (se + ppv)
  elif metric == 'fbeta':
    if beta is None:
      raise NameError(f"Weight value beta not specified for {metric}")    
    metrics = (1 + beta ** 2) * (se * ppv) / ((ppv) * (beta ** 2) + se)
    metric = f'f{beta}'
  else:
    raise ValueError(f"{metric} is not a valid metric. Valid metrics are: 'youden', 'weighted_youden', 'f1', 'fbeta'")
   
  metrics = metrics.reshape(1,-1)
  df = pd.DataFrame(metrics, index=[metric], columns=thresh_range)
  df=df.stack().reset_index()
  df.columns = ['Metric','threshold', metric]

  ax = sns.pointplot(x='threshold', y=metric,data=df)
  ax.set_xlabel('Threshold')
  ax.set_ylabel(metric)
  ax.grid(b=True, which='major', color='#d3d3d3', linewidth=1.0)
  ax.grid(b=True, which='minor', color='#d3d3d3', linewidth=0.5) 
  tick_range = np.linspace(*ax.get_xlim(), granularity)
  label_range = np.round(np.linspace(0, 1, granularity), 2)
  ax.set_xticks(tick_range)
  ax.set_xticklabels(label_range)
  
  return df.loc[df[metric].idxmax()]['threshold'] 