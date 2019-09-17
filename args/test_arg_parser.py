"""Define class for processing testing command-line arguments."""
from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        # Data args
        self.parser.add_argument('--phase',
                                 dest='data_args.phase',
                                 type=str, default='test',
                                 choices=('train', 'valid', 'test'))

        # Logger args
        self.parser.add_argument('--save_cams',
                                 dest='logger_args.save_cams',
                                 action="store_true", default=False,
                                 help=('If true, will save cams to ' +
                                       'experiment_folder/cams'))

        # Model args
        self.parser.add_argument('--config_path',
                                 dest='model_args.config_path',
                                 type=str, default=None, help='Configuration for ensemble prediction.')
