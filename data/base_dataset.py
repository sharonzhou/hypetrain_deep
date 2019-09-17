import torchvision.transforms as t
from torch.utils.data import Dataset

from constants import *



class BaseDataset(Dataset):
    def __init__(self, csv_name, is_training, transform_args):
        self.csv_name = f"{csv_name}.csv" if not csv_name.endswith(".csv") else csv_name
        self.is_training = is_training

        self._set_transforms(transform_args)

    def _set_transforms(self, transform_args):
        """Set the transforms to be applied when loading."""
        transforms_list = []

        # Shorter side scaled to transform_args.scale
        if transform_args.scale is not None:
            if transform_args.maintain_ratio:
                transforms_list += [t.Resize(transform_args.scale)]
            else:
                transforms_list += [t.Resize((transform_args.scale, transform_args.scale))]

        # Data augmentation
        if self.is_training:
            transforms_list += [t.RandomHorizontalFlip()
                                if transform_args.horizontal_flip else None,
                                t.RandomRotation(transform_args.rotate)
                                if transform_args.rotate else None,
                                t.RandomCrop((transform_args.crop,
                                              transform_args.crop))
                                if transform_args.crop else None]
        else:
            transforms_list += [t.CenterCrop((transform_args.crop,
                                              transform_args.crop))
                                if transform_args.crop else None]

        if transform_args.normalization == 'imagenet':
            normalize = t.Normalize(mean=IMAGENET_MEAN,
                                    std=IMAGENET_STD)

        transforms_list += [t.ToTensor(), normalize]

        self.transform = t.Compose([transform
                                    for transform in transforms_list
                                    if transform])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        raise NotImplementedError
