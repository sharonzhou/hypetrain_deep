import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from .point_metrics import *
from .summary_metrics import *


# Classification metric helper function
def get_classification_metrics(eval_groundtruth,
                               eval_probabilities,
                               threshold=None,
                               tune_threshold=False,
                               threshold_metric="f1",
                               seed=42,
                               logger=None):
    summary_metrics = compute_summary_metrics(eval_groundtruth,
                                              eval_probabilities,
                                              threshold=threshold,
                                              tune_threshold=tune_threshold,
                                              threshold_metric=threshold_metric,
                                              seed=seed,
                                              logger=logger)
    return summary_metrics


def get_experiment_roc_curves(experiment2probabilities, eval_groundtruth):
    """Aggregate experiment trials and return one ROC curve per experiment."""
    proc = importr('pROC')
    experiment_curves = {}
    for experiment, probabilities in experiment2probabilities.items():

        experiment_curve = proc.roc(eval_groundtruth, probabilities)
        experiment_curves[experiment] = experiment_curve

    return experiment_curves
