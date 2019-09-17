"""Define Logger class for logging information to stdout and disk."""
import sys
from tensorboardX import SummaryWriter
import time

class Logger(object):
    """Class for logging output."""
    def __init__(self, log_path, save_dir, results_dir=None):
        self.log_path = log_path
        self.log_file = log_path.open('w')

        self.tb_log_dir = save_dir / "tb"
        self.summary_writer = SummaryWriter(log_dir=self.tb_log_dir)
        
        self.results_dir = results_dir
        if results_dir is not None:
            self.metrics_path = results_dir / f"scores.txt"
            self.metrics_file = self.metrics_path.open('a')

    def log(self, *args):
        self.log_stdout(*args)
        print(*args, file=self.log_file)
        self.log_file.flush()

    def log_metrics(self, metrics, phase=None):
        t = time.time()
        if phase is not None:
            msg = f'============{phase} at {t}============'
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

    def log_stdout(self, *args):
        print(*args, file=sys.stdout)
        sys.stdout.flush()

    def close(self):
        self.log_file.close()
    
    def log_scalars(self, scalar_dict, iterations, print_to_stdout=True):
        """Log all values in a dict as scalars to TensorBoard."""
        for k, v in scalar_dict.items():
            if print_to_stdout:
                self.log_stdout(f'[{k}: {v:.3g}]')
            k = k.replace('_', '/')  # Group in TensorBoard by phase
            self.summary_writer.add_scalar(k, v, iterations)
