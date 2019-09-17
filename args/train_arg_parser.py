"""Define class for processing training command-line arguments."""
from .base_arg_parser import BaseArgParser

from constants import *


class TrainArgParser(BaseArgParser):
    """Argument parser for args used only in train mode."""
    def __init__(self):
        super(TrainArgParser, self).__init__()
        self.is_training = True

        # Model args
        self.parser.add_argument('--model',
                                 dest='model_args.model',
                                 choices=('DenseNet121', 'ResNet152',
                                          'Inceptionv4', 'ResNet18',
                                          'ResNet34', 'ResNeXt101',
                                          'SEResNeXt101', 'NASNetA',
                                          'SENet154', 'MNASNet'),
                                 default='DenseNet121',
                                 help='Model name.')
        self.parser.add_argument('--no-pretrained', dest='model_args.pretrained',
                                 action="store_false",
                                 help='Use a pretrained network.')

        # Logger args
        self.parser.add_argument('--experiment_name',
                                 dest='logger_args.experiment_name',
                                 type=str, required=True,
                                 help='Experiment name.')
        self.parser.add_argument('--iters_per_print',
                                 dest='logger_args.iters_per_print',
                                 type=int, default=16,
                                 help=('Number of iterations between ' +
                                       'printing loss to the console and ' +
                                       'TensorBoard.'))
        self.parser.add_argument('--iters_per_save',
                                 dest='logger_args.iters_per_save',
                                 type=int, default=4800,
                                 help=('Number of iterations between ' +
                                       'saving a checkpoint to save_dir.'))
        self.parser.add_argument('--iters_per_eval',
                                 dest='logger_args.iters_per_eval',
                                 type=int, default=4800,
                                 help=('Number of iterations between ' +
                                       'evaluations of the model.'))
        self.parser.add_argument('--iters_per_visual',
                                 dest='logger_args.iters_per_visual',
                                 type=int, default=4800,
                                 help=('Number of iterations between ' +
                                       'visualizing training examples.'))
        self.parser.add_argument('--max_ckpts',
                                 dest='logger_args.max_ckpts',
                                 type=int, default=10,
                                 help=('Number of checkpoints to keep ' +
                                       'before overwriting old ones.'))

        # Training args
        self.parser.add_argument('--num_epochs',
                                 dest='optim_args.num_epochs',
                                 type=int, default=5,
                                 help=('Number of epochs to train. If 0, ' +
                                       'train forever.'))
        self.parser.add_argument('--metric_name',
                                 dest='optim_args.metric_name',
                                 choices=('log_loss', 'auroc'),
                                 default='log_loss',
                                 help=('Validation metric to optimize.'))
        # Optimizer
        self.parser.add_argument('--optimizer',
                                 dest='optim_args.optimizer',
                                 type=str, default='adam',
                                 choices=('sgd', 'adam'), help='Optimizer.')
        self.parser.add_argument('--sgd_momentum',
                                 dest='optim_args.sgd_momentum',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--sgd_dampening',
                                 dest='optim_args.sgd_dampening',
                                 type=float, default=0.9,
                                 help='SGD momentum (SGD only).')
        self.parser.add_argument('--weight_decay',
                                 dest='optim_args.weight_decay',
                                 type=float, default=0.0,
                                 help='Weight decay (L2 coefficient).')
        # Learning rate
        self.parser.add_argument('--lr',
                                 dest='optim_args.lr',
                                 type=float, default=1e-4,
                                 help='Initial learning rate.')
        self.parser.add_argument('--lr_scheduler',
                                 dest='optim_args.lr_scheduler',
                                 type=str, default=None,
                                 choices=(None, 'step', 'multi_step',
                                          'plateau'),
                                 help='LR scheduler to use.')
        self.parser.add_argument('--lr_decay_gamma',
                                 dest='optim_args.lr_decay_gamma',
                                 type=float, default=0.1,
                                 help=('Multiply learning rate by this ' +
                                       'value every LR step (step and ' +
                                       'multi_step only).'))
        self.parser.add_argument('--lr_decay_step',
                                 dest='optim_args.lr_decay_step',
                                 type=int, default=100,
                                 help=('Number of epochs between each ' +
                                       'multiply-by-gamma step.'))
        self.parser.add_argument('--lr_milestones',
                                 dest='optim_args.lr_milestones',
                                 type=str, default='50,125,250',
                                 help=('Epochs to step the LR when using ' +
                                       'multi_step LR scheduler.'))
        self.parser.add_argument('--lr_patience',
                                 dest='optim_args.lr_patience',
                                 type=int, default=2,
                                 help=('Number of stagnant epochs before ' +
                                       'stepping LR.'))
        # Loss function
        self.parser.add_argument('--loss_fn',
                                 dest='optim_args.loss_fn',
                                 choices=('cross_entropy'),
                                 default='cross_entropy',
                                 help='loss function.')

        # Transform arguments
        self.parser.add_argument('--scale',
                                 dest='transform_args.scale',
                                 default=None, type=int,
                                 help="Size to scale images to.")
        self.parser.add_argument('--crop',
                                 dest='transform_args.crop',
                                 type=int, default=None,
                                 help="Size to crop images to")
        self.parser.add_argument('--normalization',
                                 dest='transform_args.normalization',
                                 default='imagenet',
                                 choices=('imagenet'),
                                 help="Values used to normalize the images.")
        self.parser.add_argument('--no-maintain_ratio',
                                 dest='transform_args.maintain_ratio',
                                 action="store_false",
                                 help=("Do not maintrain ratio of width to " +
                                       "height when scaling images."))

        # Data augmentation
        self.parser.add_argument('--rotate',
                                 dest='transform_args.rotate',
                                 type=int, default=0)
        self.parser.add_argument('--horizontal_flip',
                                 dest='transform_args.horizontal_flip',
                                 action="store_true",
                                 help=("Apply random horizontal flipping " + 
                                       "to images."))
