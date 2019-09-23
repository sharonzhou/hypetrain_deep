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

    def dense_evaluate(self, groundtruth, predictions, threshold=None, threshold_tunef1=None):

        def get_optimal_f1(groundtruth, probabilities,
                           return_threshold=False):
            """Get threshold maximizing f1 score."""
            prec, rec, threshold =\
                skm.precision_recall_curve(groundtruth,
                                           probabilities)

            f1_values = 2 * (prec * rec) / (prec + rec)

            argmax_f1 = np.nanargmax(f1_values)
            max_f1 = np.nanmax(f1_values)

            if return_threshold:
                return max_f1, threshold[argmax_f1]
            else:
                return max_f1


        def get_optimal_youdensj(groundtruth, probabilities,
                                 return_threshold=False):
            """Get threshold maximizing sensitivity + specificity."""
            fpr, tpr, threshold = skm.roc_curve(groundtruth, probabilities)

            youdens_j_values = tpr + 1 - fpr

            argmax_youdens_j = np.argmax(youdens_j_values)
            max_youdens_j = np.max(youdens_j_values)

            if return_threshold:
                return max_youdens_j, threshold[argmax_youdens_j]
            else:
                return max_youdens_j

        # Pearson's correlation coefficient r and Spearman's rank order rho
        dense_metrics = {}
        
        groundtruth = [gt[0] for gt in groundtruth.values]
        predictions = [pred[0] for pred in predictions.values]
        
        dense_metrics['pearsonr'], dense_metrics['pearsonr_pval'] = stats.pearsonr(groundtruth, predictions)
        dense_metrics['spearmanr'], dense_metrics['spearmanr_pval'] = stats.spearmanr(groundtruth, predictions)

        # AUROC on majority vote on groundtruth (binarized): auroc_dense
        binarized_groundtruth = np.array(groundtruth) > 0.5
        try:
            dense_metrics['auroc_dense'] = skm.roc_auc_score(binarized_groundtruth, predictions) 
        except ValueError:
            dense_metrics['auroc_dense'] = 0.
        try:
            dense_metrics['auprc_dense'] = skm.average_precision_score(binarized_groundtruth, predictions) 
        except ValueError:
            dense_metrics['auprc_dense'] = 0.

        # Accuracy and getting optimal threshold
        if threshold is None:
            dense_metrics['accuracy_dense'], dense_metrics['threshold_dense'] = get_optimal_youdensj(binarized_groundtruth, 
                                                                                                     predictions,
                                                                                                     return_threshold=True) 
            dense_metrics['accuracy_tunef1_dense'], dense_metrics['threshold_tunef1_dense'] = get_optimal_f1(
                                                                                                binarized_groundtruth, 
                                                                                                predictions,
                                                                                                return_threshold=True) 
        else:
            binarized_predictions = predictions > threshold
            dense_metrics['accuracy_dense'] = skm.accuracy_score(binarized_groundtruth,
                                                                 binarized_predictions)

            binarized_predictions_tunef1 = predictions > threshold_tunef1
            dense_metrics['accuracy_tunef1_dense'] = skm.accuracy_score(binarized_groundtruth,
                                                                        binarized_predictions_tunef1)
            
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
