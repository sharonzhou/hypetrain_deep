import pandas as pd
import numpy as np
import sklearn.metrics as sk_metrics
import torch.nn as nn

from aihc_stats import classification_metrics as c_metrics
from constants import *


class Evaluator(object):
    """Evaluator class for evaluating predictions against
    binary groundtruth."""
    def __init__(self,
                 logger=None,
                 phase='valid',
                 threshold=None,
                 tune_threshold=False,
                 **kwargs):

        self.logger = logger
        self.phase = phase
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
