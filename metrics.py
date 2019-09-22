#!/usr/bin/env python

import numpy as np
import pandas as pd
pd.set_option('display.max_colwidth', -1)

from functools import partial
from typing import List
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, f1_score, recall_score
from scipy import stats

def _mean_confidence_interval(data, conf=0.95, decimal=3):
  assert(conf > 0 and conf < 1), f"Confidence interval must be within (0, 1). It is {conf}"
  a = 1.0 * np.array(data)
  n = len(a)
  m, se = np.mean(a, axis=0), stats.sem(a, axis=0)
  h = se * stats.t.ppf((1 + conf) / 2., n-1)
  return np.round(m, decimal), np.round(m-h, decimal), np.round(m+h, decimal)

class MultiAvgMetrics(object):
  def __init__(self, n_classes: int, targs: List[int], preds: List[int], decimal=3) -> None:
    assert (len(targs) == len(preds)), f"Target list (length = {len(targets)}) and predictions list (length = {len(predictions)}) must be of the same length!))"
    self.targs = targs
    self.n_runs = len(self.targs)
    self.preds = preds
    self.decimal = decimal
    self.n_classes = n_classes
    
    self.cms = np.zeros((len(self.targs), n_classes, n_classes), dtype=np.int64)   
    
    for i, (targ, pred) in enumerate(zip(self.targs, self.preds)):
      self.cms[i] = confusion_matrix(targ, pred)
      
  @property
  def tps(self):
    """
      All diagonal elements of CM
    """
    tp = np.zeros((self.n_runs, self.n_classes))
    for i in range(self.n_runs):
      tp[i] = np.diag(self.cms[i])
    return tp
  
  @property
  def fns(self):
    """
      Sum of values of class's row excluding TP
    """
    fn = np.zeros((self.n_runs, self.n_classes))
    for i in range(self.n_runs):
      fn[i] = self.cms[i].sum(axis=1) - np.diag(self.cms[i])
    return fn
  
  @property
  def fps(self):
    """
      Sum of values of class's column excluding TP
    """
    fp = np.zeros((self.n_runs, self.n_classes))
    for i in range(self.n_runs):
      fp[i] = self.cms[i].sum(axis=0) - np.diag(self.cms[i])
    return fp
  
  @property
  def tns(self):
    """
      Sum of all values of excluding elements from class's row and column
    """
    tn = np.zeros((self.n_runs, self.n_classes))
    for i in range(self.n_runs):
      tn[i] = self.cms[i].sum() - (self.cms[i].sum(axis=1) + self.cms[i].sum(axis=0) - np.diag(self.cms[i]))
    return tn
  
  @property
  def prevalence_avg(self):
    return np.round(((self.fns + self.tps) / (self.tns + self.fps + self.fns + self.tps)).mean(axis=0), self.decimal)
  
  @property
  def sensitivities(self):
    return self.tps / (self.tps + self.fns)
  
  @property
  def specificities(self):
    return self.tns / (self.tns + self.fps)  
  
  @property
  def ppvs(self):
    return self.tps / (self.tps + self.fps)
  
  @property
  def npvs(self):
    return self.tns / (self.tns + self.fns)
  
  @property
  def f1s(self):
    return (2 * self.sensitivities() * self.ppvs()) / (self.sensitivities() + self.ppvs())      
  
  def sensitivity_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    if conf is not None:
      return _mean_confidence_interval(se, conf)

    return np.round(se.mean(axis=0), self.decimal,)    
  
  def specificity_avg(self, conf=None):
    sp = (self.tns / (self.tns + self.fps))
    if conf is not None:
      return _mean_confidence_interval(sp, conf)

    return np.round(sp.mean(axis=0), self.decimal)
  
  def ppv_avg(self, conf=None):
    ppv = (self.tps / (self.tps + self.fps))
    if conf is not None:
      return _mean_confidence_interval(ppv, conf)

    return np.round(ppv.mean(axis=0), self.decimal)  
  
  def npv_avg(self, conf=None):
    npv = (self.tns / (self.tns + self.fns))
    if conf is not None:
      return _mean_confidence_interval(npv, conf)

    return np.round(npv.mean(axis=0), self.decimal) 
  
  def f1_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    ppv = (self.tps / (self.tps + self.fps))
    f1 = (2 * se * ppv) / (se + ppv)
    if conf is not None:
      return _mean_confidence_interval(f1, conf)

    return np.round(f1.mean(axis=0), self.decimal)
  
  def get_class_metrics(self, class_names=None):
    if not class_names:
      class_names = list(range(self.n_classes))
      
    metrics = {
        'sensitivity': list(self.sensitivity_avg()),
        'specificity': list(self.specificity_avg()),
        'ppv': list(self.ppv_avg()),
        'npv': list(self.npv_avg()),
        'f1': list(self.f1_avg()),
      }
    
    return pd.DataFrame(metrics.values(), index=metrics.keys(), columns=class_names)
  
  def get_weighted_metrics(self):
    sensitivity = np.array([recall_score(targ, pred, average='weighted') for targ, pred in zip(self.targs, self.preds)]).mean()
    ppv = np.array([precision_score(targ, pred, average='weighted') for targ, pred in zip(self.targs, self.preds)]).mean()
    f1 = np.array([f1_score(targ, pred, average='weighted') for targ, pred in zip(self.targs, self.preds)]).mean()
    
    metrics = {
    'sensitivity': np.round((sensitivity), self.decimal),
    'ppv': np.round((ppv), self.decimal),
    'f1': np.round((f1), self.decimal),
  }
    
    return pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Value'])  

class BinaryAvgMetrics(object):
  def __init__(self, targets: List[int], predictions: List[int], probs: List[float], decimal=3) -> None:
    assert (len(targets) == len(predictions) == len(probs)), f"Target list (length = {len(targets)}), predictions list (length = {len(predictions)}) and probabilities list (length = {len(probs)}) must all be of the same length!))"
    self.targs = targets
    self.n_runs = len(self.targs)
    self.preds = predictions
    self.probs = probs
    self.decimal = 3
    
    self.cms = np.zeros((len(self.targs), 2, 2), dtype=np.int64)

    for i, (targ, pred) in enumerate(zip(self.targs, self.preds)):
      self.cms[i] = confusion_matrix(targ, pred)  

  @property
  def tns(self):
    return self.cms[:, 0, 0]
  
  @property
  def fps(self):
    return self.cms[:, 0, 1]
  
  @property
  def fns(self):
    return self.cms[:, 1, 0]
  
  @property
  def tps(self):
    return self.cms[:, 1, 1]
  
  @property
  def cm_avg(self):
    return np.ceil(np.array([[self.tns.mean(), self.fps.mean()], [self.fns.mean(), self.tps.mean()]])).astype(np.int64)
  
  @property
  def prevalence_avg(self):
    return np.round(((self.fns + self.tps) / (self.tns + self.fps + self.fns + self.tps)).mean(), self.decimal)

  def sensitivities(self):
    return self.tps / (self.tps + self.fns)
  
  def sensitivity_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    if conf is not None:
      return _mean_confidence_interval(se, conf)

    return np.round(se.mean(), self.decimal,)

  def specificities(self):
    return self.tns / (self.tns + self.fps)
  
  def specificity_avg(self, conf=None):
    sp = (self.tns / (self.tns + self.fps))
    if conf is not None:
      return _mean_confidence_interval(sp, conf)

    return np.round(sp.mean(), self.decimal)

  def ppvs(self):
    return self.tps / (self.tps + self.fps)
  
  def ppv_avg(self, conf=None):
    ppv = (self.tps / (self.tps + self.fps))
    if conf is not None:
      return _mean_confidence_interval(ppv, conf)

    return np.round(ppv.mean(), self.decimal)  

  def npvs(self):
    return self.tns / (self.tns + self.fns)
  
  def npv_avg(self, conf=None):
    npv = (self.tns / (self.tns + self.fns))
    if conf is not None:
      return _mean_confidence_interval(npv, conf)

    return np.round(npv.mean(), self.decimal)
  
  def f1s(self):
    return (2 * self.sensitivities() * self.ppvs()) / (self.sensitivities() + self.ppvs())

  def f1_avg(self, conf=None):
    se = (self.tps / (self.tps + self.fns))
    ppv = (self.tps / (self.tps + self.fps))
    f1 = (2 * se * ppv) / (se + ppv)
    if conf is not None:
      return _mean_confidence_interval(f1, conf)

    return np.round(f1.mean(), self.decimal)

  def aurocs(self):
    return np.array([roc_auc_score(targ, prob) for targ, prob in zip(self.targs, self.probs)])

  def auroc_avg(self, conf=None):
    auroc = np.array([roc_auc_score(targ, prob) for targ, prob in zip(self.targs, self.probs)])
    if conf is not None:
      return _mean_confidence_interval(auroc, conf)

    return np.round(auroc.mean(), self.decimal)

  def get_avg_metrics(self, conf=None, defn=False):
    definitions = {
      'sensitivity': "When it's ACTUALLY YES, how often does it PREDICT YES?",
      'specificity': "When it's ACTUALLY NO, how often does it PREDICT NO?",
      'ppv': "When it PREDICTS YES, how often is it correct?",
      'auroc': "Indicates how well the model is capable of distinguishing between classes",
      'npv': "When it PREDICTS NO, how often is it correct?",
      'f1': "Harmonic mean of sensitivity and ppv",
    }
    if conf is None:
      metrics = {
        'sensitivity': [self.sensitivity_avg()],
        'specificity': [self.specificity_avg()],
        'ppv': [self.ppv_avg()],
        'auroc': [self.auroc_avg()],
        'npv': [self.npv_avg()],
        'f1': [self.f1_avg()],
      }

      if defn:
        for metric, value in metrics.items():
          value.append(definitions[metric])
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Value', 'Definition'])
      else:
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Value'])

      return d

    else:
      metrics = {
        'sensitivity': [*[value for value in self.sensitivity_avg(conf)]],        
        'specificity': [*[value for value in self.specificity_avg(conf)]],
        'ppv': [*[value for value in self.ppv_avg(conf)]],
        'auroc': [*[value for value in self.auroc_avg(conf)]],   
        'npv': [*[value for value in self.npv_avg(conf)]],
        'f1': [*[value for value in self.f1_avg(conf)]],        
      }

      if defn:
        for metric, value in metrics.items():
          value.append(definitions[metric])
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Mean', 'Lower', 'Upper', 'Definition'])
      else:
        d = pd.DataFrame(metrics.values(), index=metrics.keys(), columns=['Mean', 'Lower', 'Upper'])

      return d
  
  def __repr__(self):
    s = f"Number of Runs: {self.n_runs}\n"
    return s
  
  def __len__(self):
    return len(self.targs)

def get_best_model(bam: BinaryAvgMetrics, fnames: List[str]):
  best_se, best_se_model = 0, None
  best_sp, best_sp_model = 0, None
  best_ppv, best_ppv_model = 0, None
  best_auroc, best_auroc_model = 0, None
  best_npv, best_npv_model = 0, None
  best_f1, best_f1_model = 0, None

  for i in range(bam.n_runs):
    se = bam.tps[i] / (bam.tps[i] + bam.fns[i])
    sp = bam.tns[i] / (bam.tns[i] + bam.fps[i])
    ppv = bam.tps[i] / (bam.tps[i] + bam.fps[i])
    npv = bam.tns[i] / (bam.tns[i] + bam.fns[i])
    f1 = (2 * se * ppv) / (se + ppv)

    if best_se < se:
      best_se = se
      best_se_model = fnames[i]    
    if best_sp < sp:
      best_sp = sp
      best_sp_model = fnames[i]          
    if best_ppv < ppv:
      best_ppv = ppv
      best_ppv_model = fnames[i]    
    if best_npv < npv:
      best_npv = npv
      best_npv_model = fnames[i]  
    if best_f1 < f1:
      best_f1 = f1
      best_f1_model = fnames[i]    

  for i, (targ, prob) in enumerate(zip(bam.targs, bam.probs)):
    auroc = roc_auc_score(targ, prob)
    if best_auroc < auroc:
      best_auroc = auroc
      best_auroc_model = fnames[i]

  d = {
    'sensitivity': [best_se, best_se_model],
    'specificity': [best_sp, best_sp_model],
    'ppv': [best_ppv, best_ppv_model],
    'auroc': [best_auroc, best_auroc_model],
    'npv': [best_npv, best_npv_model],
    'f1': [best_f1, best_f1_model],
  }

  return pd.DataFrame(d.values(), index=d.keys(), columns=['Value', 'Model File'])
