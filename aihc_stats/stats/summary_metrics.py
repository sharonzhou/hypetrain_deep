import numpy as np
import sklearn.metrics as skm
from rpy2.robjects.packages import importr
from collections import defaultdict

from .bootstrap import Bootstrapper


def roc_auc_score(groundtruth, probabilities):
    return skm.roc_auc_score(groundtruth, probabilities)

def pr_auc_score(groundtruth, probabilities):
    return skm.average_precision_score(groundtruth, probabilities)
    
def log_loss_score(groundtruth, probabilities):
    return skm.log_loss(groundtruth, probabilities)

def f1_score(groundtruth, probabilities, threshold):
    predictions = probabilities > threshold
    return skm.f1_score(groundtruth, predictions)

def precision_score(groundtruth, probabilities, threshold):
    predictions = probabilities > threshold
    return skm.precision_score(groundtruth, predictions)

def recall_score(groundtruth, probabilities, threshold):
    predictions = probabilities > threshold
    return skm.recall_score(groundtruth, predictions)

def accuracy_score(groundtruth, probabilities, threshold):
    predictions = probabilities > threshold
    return skm.accuracy_score(groundtruth, predictions)


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


def compute_summary_metrics(eval_groundtruth,
                            eval_probabilities,
                            threshold, # None, except for test, put best threshold found on valid
                            tune_threshold, # True for valid, False for test
                            threshold_metric, # f1 or youdensj
                            seed=42,
                            logger=None):
    """Compute summary metrics."""
    summary_metric_names = ["roc_auc", "pr_auc", "log_loss"]
    summary_metrics = {}
    global_funcs = globals()

    for summary_metric_name in summary_metric_names:
        summary_metric_func = global_funcs[f"{summary_metric_name}_score"]
        summary_metrics[summary_metric_name] = summary_metric_func(eval_groundtruth,
                                                                   eval_probabilities)
    # Evaluate best threshold for accuracy
    if tune_threshold and threshold_metric == "f1":
        _, threshold = get_optimal_f1(eval_groundtruth, eval_probabilities, return_threshold=True)
        summary_metrics['threshold_f1'] = threshold
    elif tune_threshold:
        _, threshold = get_optimal_youdensj(eval_groundtruth, eval_probabilities, return_threshold=True)
        summary_metrics['threshold_youdensj'] = threshold

    # Compute threshold metrics
    if threshold is not None:
        summary_metrics['threshold'] = threshold

        # Use threshold on remaining scores
        thres_summary_metric_names = ["f1", "precision", "recall", "accuracy"]
        for thres_summary_metric_name in thres_summary_metric_names:
            thres_summary_metric_func = global_funcs[f"{thres_summary_metric_name}_score"]
            summary_metrics[thres_summary_metric_name] = thres_summary_metric_func(eval_groundtruth,
                                                                                   eval_probabilities,
                                                                                   threshold)

    if logger is not None:
        logger.log(f"Summary metrics: {summary_metrics}")

    return summary_metrics


# Curves -- globals
roc_curve = skm.roc_curve
pr_curve = skm.precision_recall_curve


# Curve helper function
def get_curves(groundtruth,
               probabilities,
               logger=None):
    roc = roc_curve(groundtruth, probabilities)
    pr = pr_curve(groundtruth, probabilities)

    curves = {}
    curves["roc"] = roc
    curves["pr"] = pr

    return curves
