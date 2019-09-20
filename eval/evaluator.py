import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
import torch.nn as nn
from scipy import stats
import sklearn.metrics as skm

from aihc_stats import classification_metrics as c_metrics
from constants import *


class Evaluator(object):
    """Evaluator class for evaluating predictions against
    binary groundtruth."""
    def __init__(self,
                 logger=None,
                 threshold=None,
                 tune_threshold=False,
                 **kwargs):

        self.logger = logger
        self.threshold = threshold
        self.tune_threshold = tune_threshold

    def evaluate(self, groundtruth, predictions):
        summary_metrics =\
            c_metrics.get_classification_metrics(groundtruth.values,
                                                 predictions.values,
                                                 self.threshold,
                                                 self.tune_threshold,
                                                )

        return {**summary_metrics}

    def dense_evaluate(self, groundtruth, predictions):
        # Pearson's correlation coefficient r and Spearman's rank order rho
        dense_metrics = {}
        
        groundtruth = [gt[0] for gt in groundtruth.values]
        predictions = [pred[0] for pred in predictions.values]
        
        dense_metrics['pearsonr'], dense_metrics['pearsonr_pval'] = stats.pearsonr(groundtruth, predictions)
        dense_metrics['spearmanr'], dense_metrics['spearmanr_pval'] = stats.spearmanr(groundtruth, predictions)

        # AUROC on majority vote on groundtruth (binarized): auroc_dense
        binarized_groundtruth = np.array(groundtruth) > 0.5
        dense_metrics['auroc_dense'] = skm.roc_auc_score(binarized_groundtruth, predictions) 
        dense_metrics['auprc_dense'] = skm.average_precision_score(binarized_groundtruth, predictions) 

        return {**dense_metrics}

    def get_loss_fn(self, loss_fn_name):
        """Get the loss function used for training.

        Args:
            loss_fn_name: Name of loss function to use.
        """
        if loss_fn_name == "cross_entropy":
            loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Loss fn {loss_fn} not supported.")

        return loss_fn
