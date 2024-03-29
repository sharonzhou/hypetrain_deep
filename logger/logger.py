"""Define Logger class for logging information to stdout and disk."""
import sys
from tensorboardX import SummaryWriter
import time
import csv
from pathlib import Path

from constants import *

class Logger(object):
    """Class for logging output."""
    def __init__(self, logger_args, data_args=None, optim_args=None, test_args=None):
        self.log_path = logger_args.log_path
        self.log_file = self.log_path.open('w')

        self.tb_log_dir = Path(logger_args.tb_dir)
        self.summary_writer = SummaryWriter(log_dir=self.tb_log_dir)
        
        # Logger args
        self.results_dir = None
        if 'results_dir' in logger_args:
            self.results_dir = logger_args.results_dir
            self.metrics_path = Path(self.results_dir) / f"scores.txt"
            self.metrics_file = self.metrics_path.open('a')

        # Data args
        self.models = data_args.models if data_args is not None else None
        self.models_valid = data_args.models_valid if data_args is not None else None

        # Optim args
        self.metric_name = optim_args.metric_name if optim_args is not None else None
    
        # Test args
        self.models_test = test_args.models_test if test_args is not None else None
        self.final_csv = test_args.final_csv if test_args is not None else False
        if self.final_csv:
            self.final_csv_path = PROJECT_DIR / f"final_scores_all.csv"
            self.final_csv_file = self.final_csv_path.open('a')
            self.final_csv_writer = csv.writer(self.final_csv_file, delimiter=',', lineterminator='\n')
            
            self.final_csv_abbrev_path = PROJECT_DIR / f"final_scores.csv"
            self.final_csv_abbrev_file = self.final_csv_abbrev_path.open('a')
            self.final_csv_abbrev_writer = csv.writer(self.final_csv_abbrev_file, delimiter=',', lineterminator='\n')

            self.final_csv_dense_path = PROJECT_DIR / f"final_dense_scores_all.csv"
            self.final_csv_dense_file = self.final_csv_dense_path.open('a')
            self.final_csv_dense_writer = csv.writer(self.final_csv_dense_file, delimiter=',', lineterminator='\n')

            self.final_csv_dense_abbrev_path = PROJECT_DIR / f"final_dense_scores.csv"
            self.final_csv_dense_abbrev_file = self.final_csv_dense_abbrev_path.open('a')
            self.final_csv_dense_abbrev_writer = csv.writer(self.final_csv_dense_abbrev_file, delimiter=',', lineterminator='\n')


    def log(self, *args):
        self.log_stdout(*args)
        print(*args, file=self.log_file)
        self.log_file.flush()

    def log_metrics(self, metrics, phase=None, pretrained=True):
        t = time.time()
        if phase is not None:
            msg = f'============{phase} at {t}============'
            if self.models is not None:
                if phase == 'valid' and self.models_valid is not None:
                    msg += f'\ntrained on {self.models}, validated on {self.models_valid}'
                elif phase == 'test' and self.models_test is not None:
                    msg += f'\ntrained on {self.models}, tested on {self.models_test}'
        else:
            msg = '=============={t}============='
        if self.results_dir is not None:
            self.log_stdout(msg)
            print(msg, file=self.metrics_file)
            self.metrics_file.flush()
        else:
            self.log(f"[{msg}]")

        for metric, value in metrics.items():
            msg = f'{metric}:\t{value}'
            if self.results_dir is not None:
                self.log_stdout(msg)
                print(msg, file=self.metrics_file)
                self.metrics_file.flush()
            else:
                self.log(f"[{msg}]")

        if self.final_csv and 'test' in phase and self.models is not None and self.models_test is not None and self.metric_name is not None:
            trained_on = ''
            for m in self.models:
                if m == 'began':
                    trained_on += 'b'
                elif m == 'progan':
                    trained_on += 'p'
                elif m == 'wgan_gp':
                    trained_on += 'w'
                elif m == 'stylegan':
                    trained_on += 's'
            tested_on = ''
            for m in self.models_test:
                if m == 'began':
                    tested_on += 'b'
                elif m == 'progan':
                    tested_on += 'p'
                elif m == 'wgan_gp':
                    tested_on += 'w'
                elif m == 'stylegan':
                    tested_on += 's'
            if phase == 'test':
                row = [t, trained_on, tested_on, self.metric_name, pretrained, metrics['threshold'], metrics['accuracy'], metrics['precision'], metrics['recall'], metrics['f1'], metrics['roc_auc'], metrics['pr_auc'], metrics['log_loss']]
                self.final_csv_writer.writerow(row)
                print(f'Appended all metrics to {self.final_csv_path}')

                row_abbrev = [t, trained_on, tested_on, self.metric_name, pretrained, metrics['accuracy']]
                self.final_csv_abbrev_writer.writerow(row_abbrev)
                print(f'Appended salient metrics to {self.final_csv_abbrev_path}')
            elif phase == 'dense_test':
                row = [t, trained_on, tested_on, self.metric_name, pretrained, metrics['auroc_dense'], metrics['auprc_dense'], metrics['pearsonr'], metrics['pearsonr_pval'], metrics['spearmanr'], metrics['spearmanr_pval']]
                self.final_csv_dense_writer.writerow(row)
                print(f'Appended all dense metrics to {self.final_csv_dense_path}')
                
                row_abbrev = [t, trained_on, tested_on, self.metric_name, pretrained, metrics['auroc_dense'], metrics['auprc_dense'], metrics['pearsonr'], metrics['spearmanr']]
                self.final_csv_dense_abbrev_writer.writerow(row_abbrev)
                print(f'Appended salient dense metrics to {self.final_csv_dense_abbrev_path}')

    def log_stdout(self, *args):
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def close(self):
        self.log_file.close()
    
    def log_scalars(self, scalar_dict, iterations, phase, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.log_stdout(f'[{k}: {v:.3g}]')
            k = f'{phase}/{k}' # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, iterations)
