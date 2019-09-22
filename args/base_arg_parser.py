"""Define base class for processing command-line arguments."""
import argparse
import copy
import json
import torch
import torch.backends.cudnn as cudnn
from pathlib import Path

from constants import *


class BaseArgParser(object):
    """Base argument parser for args shared between test and train modes."""
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='CXR')

        # Logger args
        self.parser.add_argument('--save_dir',
                                 dest='logger_args.save_dir',
                                 type=str, default=str(SAVE_MODEL_DIR),
                                 help='Directory to save model data.')

        # Data args
        self.parser.add_argument('--csv_name',
                                 dest='data_args.csv_name',
                                 default=None,
                                 help=("Replace csv name (one of " +
                                       "train, valid, test by default) " +
                                       "with this string."))
        self.parser.add_argument('--batch_size',
                                 dest='data_args.batch_size',
                                 type=int, default=16,
                                 help='Batch size for training / evaluation.')
        self.parser.add_argument('--toy',
                                 dest='data_args.toy',
                                 action="store_true",
                                 help='Use toy dataset.')
        self.parser.add_argument('--dataset',
                                 dest='data_args.dataset',
                                 type=str, default='usgs',
                                 help=('Name of dataset. Directories ' +
                                       'are specified in constants.'))
        self.parser.add_argument('--models',
                                 dest='data_args.models',
                                 type=str, 
                                 default='began,wgan_gp,progan,stylegan',
                                 help=('Models to use data for.'))
        self.parser.add_argument('--models_valid',
                                 dest='data_args.models_valid',
                                 type=str, 
                                 default='began,wgan_gp,progan,stylegan',
                                 help=('Models to use data for - on validation.'))
        self.parser.add_argument('--num_workers',
                                 dest='data_args.num_workers',
                                 type=int, default=8,
                                 help='Number of threads for the DataLoader.')

        # Run args
        self.parser.add_argument('--gpu_ids',
                                 type=str, default='0',
                                 help=('Comma-separated list of GPU IDs. ' +
                                       'Use -1 for CPU.'))

        self.is_training = None

    def namespace_to_dict(self, args):
        """Turns a nested Namespace object to a nested dictionary"""
        args_dict = vars(copy.deepcopy(args))

        for arg in args_dict:
            obj = args_dict[arg]
            if isinstance(obj, argparse.Namespace):
                args_dict[arg] = self.namespace_to_dict(obj)

        return args_dict

    def fix_nested_namespaces(self, args):
        """Makes sure that nested namespaces work
            Args:
                args: argsparse.Namespace object containing all the arguments
            e.g args.data_args.batch_size

            Obs: Only one level of nesting is supported.
        """
        group_name_keys = []

        for key in args.__dict__:
            if '.' in key:
                group, name = key.split('.')
                group_name_keys.append((group, name, key))

        for group, name, key in group_name_keys:
            if group not in args:
                args.__dict__[group] = argparse.Namespace()

            args.__dict__[group].__dict__[name] = args.__dict__[key]
            del args.__dict__[key]

    def args_to_list(self, csv, allow_empty, arg_type=int, allow_negative=True):
        """Convert comma-separated arguments to a list.

        Args:
            csv: Comma-separated list of arguments as a string.
            allow_empty: If True, allow the list to be empty. Otherwise return None instead of empty list.
            arg_type: Argument type in the list.
            allow_negative: If True, allow negative inputs.

        Returns:
            List of arguments, converted to `arg_type`.
        """
        arg_vals = [arg_type(d) for d in str(csv).split(',')]
        if not allow_negative and arg_type == int:
            arg_vals = [v for v in arg_vals if v >= 0]
        if not allow_empty and len(arg_vals) == 0:
            return None
        return arg_vals

    def parse_args(self):
        """Parse command-line arguments and set up directories and other run
        args for training and testing."""
        self.parser.set_defaults(**{"model_args.pretrained": True,
                                    "transform_args.maintain_ratio": True})

        args = self.parser.parse_args()

        # Make args a nested Namespace
        self.fix_nested_namespaces(args)

        if self.is_training:
            log_name = "train_log.txt"

            # Set up model save directory for logging.
            save_dir = Path(args.logger_args.save_dir) /\
                args.logger_args.experiment_name
            args_save_dir = save_dir

            tb_dir = PROJECT_DIR / "tb" / args.logger_args.experiment_name

        else:
            args.test_args.models_test = self.args_to_list(args.test_args.models_test,
                                                           allow_empty=False,
                                                           arg_type=str)

            log_name = f"{args.test_args.phase}_log.txt"

            # Obtain save dir from ckpt path.
            save_dir = Path(args.test_args.ckpt_path).parent
            args.logger_args.experiment_name = save_dir.name

            # Make directory to save results.
            results_dir = save_dir / "results" / args.test_args.phase
            results_dir.mkdir(parents=True, exist_ok=True)
            args.logger_args.results_dir = str(results_dir)

            args_save_dir = results_dir
            
            tb_name = args.logger_args.experiment_name + '-' + args.test_args.phase
            tb_dir = PROJECT_DIR / "tb" / tb_name

        tb_dir.mkdir(parents=True, exist_ok=True)
        args.logger_args.tb_dir = str(tb_dir)

        # Create the model save directory.
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save args to a JSON file in the args save directory.
        args_path = args_save_dir / 'args.json'
        with open(args_path, 'w') as fh:
            args_dict = self.namespace_to_dict(args)
            json.dump(args_dict, fh, indent=4,
                      sort_keys=True)
            fh.write('\n')

        args.logger_args.save_dir = save_dir
        args.logger_args.log_path = args.logger_args.save_dir / log_name

        # Add configuration flags outside of the CLI
        args.is_training = self.is_training
        if args.is_training:
            iters_per_eval = args.logger_args.iters_per_eval
            if iters_per_eval % args.data_args.batch_size != 0:
                raise ValueError('iters_per_eval must be divisible ' +
                                 'by batch_size.')
            elif args.logger_args.iters_per_save % iters_per_eval != 0:
                raise ValueError('iters_per_save must be divisible ' +
                                 'by iters_per_eval.')

            if 'loss' in args.optim_args.metric_name:
                args.optim_args.maximize_metric = False
            elif args.optim_args.metric_name in ['accuracy', 'auroc_dense', 'auroc', 'auprc_dense', 'auprc', 'pearsonr', 'spearmanr', 'precision', 'recall', 'f1']:
                args.optim_args.maximize_metric = True
            else:
                raise ValueError(f"metric {args.optim_args.metric_name} " + 
                                 "not supported.")

            # Set start epoch. Note: this gets updated if we load a checkpoint
            args.optim_args.start_epoch = 1

        # Set up available GPUs
        args.gpu_ids = self.args_to_list(args.gpu_ids, allow_empty=True,
                                         arg_type=int, allow_negative=False)
        if len(args.gpu_ids) > 0 and torch.cuda.is_available():
            # Set default GPU for `tensor.to('cuda')`
            torch.cuda.set_device(args.gpu_ids[0])
            cudnn.benchmark = True
            args.device = 'cuda'
        else:
            args.device = 'cpu'
        
        # Listify models to train, val, test on
        args.data_args.models = self.args_to_list(args.data_args.models,
                                                  allow_empty=False,
                                                  arg_type=str)
        args.data_args.models_valid = self.args_to_list(args.data_args.models_valid,
                                                        allow_empty=False,
                                                        arg_type=str)
        return args
