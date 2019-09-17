import torch.utils.data as data

from .dataset import Dataset
from constants import *


def get_loader(phase, data_args, transform_args,
               is_training, return_info_dict,
               logger=None):
    """Get PyTorch data loader.

    Args:
        phase: string name of training phase {train, valid, test}.
        data_args: Namespace of data arguments.
        transform_args: Namespace of transform arguments.
        is_training: Bool indicating whether in training mode.
        return_info_dict: Bool indicating whether to return extra info
                          in batches.
        logger: Optional Logger object for printing data to stdout and file.

    Return:
        loader: PyTorch DataLoader object
    """
    shuffle = is_training

    csv_name = phase if data_args.csv_name is None\
                     else data_args.csv_name

    # Instantiate the Dataset class.
    dataset = Dataset(phase, csv_name, is_training, transform_args,
                      data_args.toy, logger, models=data_args.models)

    loader = data.DataLoader(dataset,
                             batch_size=data_args.batch_size,
                             shuffle=shuffle,
                             num_workers=data_args.num_workers)

    return loader
