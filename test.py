"""Entry-point script to train models."""
import torch

from args import TestArgParser
from logger import Logger
from predict import Predictor, EnsemblePredictor
from saver import ModelSaver
from data import get_loader
from eval import Evaluator
from constants import *


def test(args):
    """Run model testing."""
    test_args = args.test_args
    logger_args = args.logger_args
    
    # Load the model at ckpt_path.
    ckpt_path = test_args.ckpt_path
    ckpt_save_dir = Path(ckpt_path).parent

    # Get model args from checkpoint and add them to
    # command-line specified model args.
    model_args, data_args, optim_args, logger_args\
        = ModelSaver.get_args(cl_logger_args=logger_args,
                              ckpt_save_dir=ckpt_save_dir)

    model, ckpt_info = ModelSaver.load_model(ckpt_path=ckpt_path,
                                             gpu_ids=args.gpu_ids,
                                             model_args=model_args,
                                             is_training=False)


    # Get logger.
    logger = Logger(logger_args, optim_args, test_args)
    
    # Instantiate the Predictor class for obtaining model predictions.
    predictor = Predictor(model=model, device=args.device)

    phase = test_args.phase
    is_test = False
    if phase == 'test':
        is_test = True
        phase = 'valid' # Run valid first to get threshold

    print(f"======================{phase}=======================")
    # Get phase loader object.
    loader = get_loader(phase=phase,
                        data_args=data_args,
                        transform_args=transform_args,
                        is_training=False,
                        logger=logger)
    # Obtain model predictions.
    predictions, groundtruth = predictor.predict(loader)

    # Instantiate the evaluator class for evaluating models.
    evaluator = Evaluator(logger=logger,
                          tune_threshold=True)

    # Get model metrics and curves on the phase dataset.
    metrics = evaluator.evaluate(groundtruth, predictions)
    
    # Log metrics to stdout and file.
    logger.log_stdout(f"Writing metrics to {logger.metrics_path}.")
    logger.log_metrics(metrics, phase=phase)

    if is_test:
        phase = 'test'
        threshold = metrics['threshold']
        print(f"======================{phase}=======================")

        # Get phase loader object.
        loader = get_loader(phase=phase,
                            data_args=data_args,
                            transform_args=transform_args,
                            is_training=False,
                            logger=logger)
        # Obtain model predictions.
        predictions, groundtruth = predictor.predict(loader)

        # Instantiate the evaluator class for evaluating models.
        evaluator = Evaluator(logger=logger,
                              threshold=threshold,
                              tune_threshold=False)

        # Get model metrics and curves on the phase dataset.
        metrics = evaluator.evaluate(groundtruth, predictions)
        # Log metrics to stdout and file.
        logger.log_stdout(f"Writing metrics to {logger.metrics_path}.")
        logger.log_metrics(metrics, phase=phase)

        # Dense test
        phase = 'dense_test'
        dense_data_args = data_args
        dense_data_args.csv_name = f'{phase}.csv'
        dense_loader = get_loader(phase=phase,
                                  data_args=data_args,
                                  transform_args=transform_args,
                                  is_training=False,
                                  logger=logger)
        dense_predictions, dense_groundtruth = predictor.predict(dense_loader)
        dense_metrics = evaluator.dense_evaluate(dense_groundtruth, dense_predictions)
        logger.log_stdout(f"Writing metrics to {logger.metrics_path}.")
        logger.log_metrics(dense_metrics, phase=phase)


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy('file_system')
    parser = TestArgParser()
    test(parser.parse_args())
