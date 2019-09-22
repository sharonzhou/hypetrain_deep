"""Define model saver class."""
import copy
import json
import queue
import shutil
import torch
import torch.nn as nn
from argparse import Namespace

import models
from constants import *


class ModelSaver(object):
    """Class to save and load model ckpts."""
    def __init__(self, save_dir, iters_per_save, max_ckpts,
                 metric_name='val_loss', maximize_metric=False,
                 keep_topk=True, logger=None, **kwargs):
        """
        Args:
            save_dir: Directory to save checkpoints.
            iters_per_save: Number of iterations between each save.
            max_ckpts: Maximum number of checkpoints to keep before
                       overwriting old ones.
            metric_name: Name of metric used to determine best model.
            maximize_metric: If true, best checkpoint is that which
                             maximizes the metric value passed in via save.
            If false, best checkpoint minimizes the metric.
            keep_topk: Keep the top K checkpoints, rather than the most
                       recent K checkpoints.
        """
        super(ModelSaver, self).__init__()

        self.save_dir = save_dir
        self.iters_per_save = iters_per_save
        self.max_ckpts = max_ckpts
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_metric_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.keep_topk = keep_topk
        self.logger = logger

    def _is_best(self, metric_val):
        """Check whether metric_val is the best one we've seen so far."""
        if metric_val is None:
            return False
        return (self.best_metric_val is None
                or (self.maximize_metric and
                    self.best_metric_val < metric_val)
                or (not self.maximize_metric and
                    self.best_metric_val > metric_val))

    def save(self, iteration, epoch, model, optimizer, device, metric_val):
        """Save model parameters to disk.

        Args:
            iteration: Iteration that just finished.
            epoch: epoch to stamp on the checkpoint
            model: Model to save.
            optimizer: Optimizer for model parameters.
            device: Device where the model/optimizer parameters belong.
            metric_val: Value for determining whether checkpoint
                        is best so far.
        """
        lr_scheduler = None if optimizer.lr_scheduler is None\
            else optimizer.lr_scheduler.state_dict()
        ckpt_dict = {
            'ckpt_info': {'epoch': epoch, 'iteration': iteration,
                          self.metric_name: metric_val},
            'model_name': model.module.__class__.__name__,
            'model_state': model.to('cpu').state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler
        }
        model.to(device)

        ckpt_path = self.save_dir / f'iter_{iteration}.pth.tar'
        if self.logger is not None:
            self.logger.log(f"Saving model to {ckpt_path}.")
        torch.save(ckpt_dict, ckpt_path)

        if self._is_best(metric_val):
            # Save the best model
            best_path = self.save_dir / 'best.pth.tar'
            if self.logger is not None:
                self.logger.log("Saving the model based on metric=" +
                                f"{self.metric_name} and maximize=" +
                                f"{self.maximize_metric} with value" +
                                f"={metric_val} to {best_path}.")
            shutil.copy(ckpt_path, best_path)
            self.best_metric_val = metric_val

        # Add checkpoint path to priority queue (lower priority order gets
        # removed first)
        if not self.keep_topk:
            priority_order = iteration
        elif self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, ckpt_path))

        # Remove a checkpoint if more than max_ckpts ckpts saved
        if self.ckpt_paths.qsize() > self.max_ckpts:
            _, oldest_ckpt = self.ckpt_paths.get()
            try:
                oldest_ckpt.unlink()
            except OSError:
                pass

    @classmethod
    def load_ckpt_args(cls, ckpt_save_dir):
        """Load args from model ckpt.

        Args:
            ckpt_save_dir: pathlib directory pointing to model args.

        Returns:
            transform_args: Namespace of transform arguments
                            read from ckpt_path.
        """
        ckpt_args_path = ckpt_save_dir / 'args.json'
        with open(ckpt_args_path) as f:
            ckpt_args = json.load(f)

        ckpt_model_args = ckpt_args['model_args']
        # Delete ckpt_path used to instantiate the model during training
        # so that it does not overwrite the ckpt_path provided during testing.
        del ckpt_model_args['ckpt_path']     
        ckpt_data_args = ckpt_args['data_args']
        ckpt_optim_args = ckpt_args['optim_args']
        ckpt_logger_args = ckpt_args['logger_args']
        return (Namespace(**ckpt_model_args),
                Namespace(**ckpt_data_args),
                Namespace(**ckpt_optim_args),
                Namespace(**ckpt_logger_args))

    @classmethod
    def get_args(cls, cl_logger_args, ckpt_save_dir):
        """Read args from ckpt_save_dir and make a new namespace combined with
        model_args from the command line."""
        logger_args = copy.deepcopy(cl_logger_args)
        ckpt_model_args, ckpt_data_args, ckpt_optim_args, ckpt_logger_args =\
            cls.load_ckpt_args(ckpt_save_dir)
        logger_args.__dict__.update(ckpt_logger_args.__dict__)
        
        if 'models' in ckpt_data_args:
            ckpt_data_args.models = [d for d in ckpt_data_args.models.split(',')]
        if 'models_valid' in ckpt_data_args:
            ckpt_data_args.models_valid = [d for d in ckpt_data_args.models_valid.split(',')]
        if 'models_test' in ckpt_data_args:
            ckpt_data_args.models_test = [d for d in ckpt_data_args.models_test.split(',')]

        return ckpt_model_args, ckpt_data_args, ckpt_optim_args, logger_args

    @classmethod
    def load_model(cls, ckpt_path, gpu_ids, model_args,
                   is_training=False):
        """Load model parameters from disk.

        Args:
            ckpt_path: Path to checkpoint to load.
            gpu_ids: GPU IDs for DataParallel.
            model_args: Model arguments to instantiate the model object.
            is_training: Bool indicating if training the model.
    
        Returns:
            Model loaded from checkpoint, dict of additional
            checkpoint info (e.g. epoch, metric).
        """
        device = f'cuda:{gpu_ids[0]}' if len(gpu_ids) > 0 else 'cpu'
        ckpt_dict = torch.load(ckpt_path, map_location=device)

        # Build model, load parameters
        model_fn = models.__dict__[ckpt_dict['model_name']]

        model = model_fn(model_args)
        model = nn.DataParallel(model, gpu_ids)
        model_dict = ckpt_dict['model_state']

        model.load_state_dict(model_dict)

        model = model.to(device)

        if is_training:
            model.train()
        else:
            model.eval()

        return model, ckpt_dict['ckpt_info']
