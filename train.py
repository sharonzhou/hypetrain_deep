"""Entry-point script to train models."""
import torch
import torch.nn as nn

import models
from args import TrainArgParser
from logger import Logger
from saver import ModelSaver
from predict import Predictor
from data import get_loader
from eval import Evaluator
from optim import Optimizer
from constants import *


def train(args):
    """Run model training."""

    # Get nested namespaces.
    model_args = args.model_args
    logger_args = args.logger_args
    optim_args = args.optim_args
    data_args = args.data_args
    transform_args = args.transform_args

    # Get logger.
    logger = Logger(logger_args.log_path, logger_args.save_dir)

    if model_args.ckpt_path:
        # CL-specified args are used to load the model, rather than the
        # ones saved to args.json.
        model_args.pretrained = False
        ckpt_path = model_args.ckpt_path
        assert False
        model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                                 gpu_ids=args.gpu_ids,
                                                 model_args=model_args,
                                                 is_training=True)
        optim_args.start_epoch = ckpt_info['epoch'] + 1
    else:
        # If no ckpt_path is provided, instantiate a new randomly
        # initialized model.
        model_fn = models.__dict__[model_args.model]
        model = model_fn(model_args)
        model = nn.DataParallel(model, args.gpu_ids)
    # Put model on gpu or cpu and put into training mode.
    model = model.to(args.device)
    model.train()


    # Get train and valid loader objects.
    train_loader = get_loader(phase="train",
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=True,
                              logger=logger)
    valid_loader = get_loader(phase="valid",
                              data_args=data_args,
                              transform_args=transform_args,
                              is_training=False,
                              logger=logger)
    dense_valid_loader = get_loader(phase="dense_valid",
                                    data_args=data_args,
                                    transform_args=transform_args,
                                    is_training=False,
                                    logger=logger)

    # Instantiate the predictor class for obtaining model predictions.
    predictor = Predictor(model, args.device)

    # Instantiate the evaluator class for evaluating models.
    # By default, get best performance on validation set.
    evaluator = Evaluator(logger=logger,
                          tune_threshold=True)

    # Instantiate the saver class for saving model checkpoints.
    saver = ModelSaver(save_dir=logger_args.save_dir,
                       iters_per_save=logger_args.iters_per_save,
                       max_ckpts=logger_args.max_ckpts,
                       metric_name=optim_args.metric_name,
                       maximize_metric=optim_args.maximize_metric,
                       keep_topk=True,
                       logger=logger)

    # Instantiate the optimizer class for guiding model training.
    optimizer = Optimizer(parameters=model.parameters(),
                          optim_args=optim_args,
                          batch_size=data_args.batch_size,
                          iters_per_print=logger_args.iters_per_print,
                          iters_per_visual=logger_args.iters_per_visual,
                          iters_per_eval=logger_args.iters_per_eval,
                          dataset_len=len(train_loader.dataset),
                          logger=logger)
    if model_args.ckpt_path:
        # Load the same optimizer as used in the original training.
        optimizer.load_optimizer(ckpt_path=model_args.ckpt_path,
                                 gpu_ids=args.gpu_ids)

    loss_fn = evaluator.get_loss_fn(loss_fn_name=optim_args.loss_fn)
    
    # Run training
    while not optimizer.is_finished_training():
        optimizer.start_epoch()

        for inputs, targets in train_loader:
            optimizer.start_iter()

            if optimizer.global_step % optimizer.iters_per_eval == 0:
                # Only evaluate every iters_per_eval examples.
                predictions, groundtruth = predictor.predict(valid_loader)
                metrics = evaluator.evaluate(groundtruth, predictions)

                if optim_args.metric_name in ['pearsonr', 'spearmanr']:
                    dense_predictions, dense_groundtruth = predictor.predict(dense_valid_loader)
                    dense_metrics = evaluator.dense_evaluate(dense_groundtruth, dense_predictions)
                    # Merge the metrics dicts together
                    metrics = {**metrics, **dense_metrics}
                
                # Log metrics to stdout.
                logger.log_metrics(metrics, phase='train')

                if optimizer.global_step % logger_args.iters_per_save == 0:
                    # Only save every iters_per_save examples directly
                    # after evaluation.
                    saver.save(iteration=optimizer.global_step,
                               epoch=optimizer.epoch,
                               model=model,
                               optimizer=optimizer,
                               device=args.device,
                               metric_val=metrics[optim_args.metric_name])

                # Step learning rate scheduler.
                optimizer.step_scheduler(metrics[optim_args.metric_name])

            with torch.set_grad_enabled(True):

                # Run the minibatch through the model.
                logits = model(inputs.to(args.device))

                # Compute the minibatch loss.
                loss = loss_fn(logits, targets.to(args.device))

                # Log the data from this iteration.
                optimizer.log_iter(inputs, logits, targets, loss)

                # Perform a backward pass.
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            optimizer.end_iter()

        optimizer.end_epoch(metrics)

    # Save the most recent model.
    saver.save(iteration=optimizer.global_step,
               epoch=optimizer.epoch,
               model=model,
               optimizer=optimizer,
               device=args.device,
               metric_val=metrics[optim_args.metric_name])


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TrainArgParser()
    train(parser.parse_args())
