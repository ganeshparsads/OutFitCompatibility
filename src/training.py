import logging
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from sklearn import metrics
from torch.optim import lr_scheduler
from torchvision import models
import matplotlib.pyplot as plt
import pandas as pd
# from sklearn.metrics import roc_curve

def plot_roc_and_auc(y_test, y_preds):
  fpr, tpr, thresholds = metrics.roc_curve(y_test, y_preds)
  plt.figure(1)
  plt.plot([0, 1], [0, 1], 'y--')
  plt.plot(fpr, tpr, marker='.')
  plt.xlabel('False positive rate')
  plt.ylabel('True positive rate')
  plt.title('ROC curve')
  plt.show()

  i = np.arange(len(tpr)) 
  roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'thresholds' : pd.Series(thresholds, index=i)})
  ideal_roc_thresh = roc.iloc[(roc.tf-0).abs().argsort()[:1]]  #Locate the point where the value is close to 0
  print("Ideal threshold is: ", ideal_roc_thresh['thresholds'])

  auc_value = metrics.auc(fpr, tpr)
  print("Area under curve, AUC = ", auc_value)

  return auc_value

