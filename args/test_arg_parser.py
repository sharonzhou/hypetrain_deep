"""Define class for processing testing command-line arguments."""
from .base_arg_parser import BaseArgParser


class TestArgParser(BaseArgParser):
    """Argument parser for args used only in test mode."""
    def __init__(self):
        super(TestArgParser, self).__init__()
        self.is_training = False

        # Test args
        self.parser.add_argument('--phase',
                                 dest='test_args.phase',
                                 type=str, default='test',
                                 choices=('train', 'valid', 'test'))
        self.parser.add_argument('--final_csv',
                                 dest='test_args.final_csv',
                                 action='store_true', default=False,
                                 help='Save scores to final csv.')
        self.parser.add_argument('--ckpt_path',
                                 dest='test_args.ckpt_path',
                                 type=str, required=True,
                                 help='Checkpoint path for tuning.')
