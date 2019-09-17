import cv2
import torch
import pandas as pd
import random
from PIL import Image

from .base_dataset import BaseDataset
from constants import *


class Dataset(BaseDataset):
    def __init__(self, phase, csv_name, is_training, transform_args,
                 toy, logger, models=["began", "wgan_gp", "progan", "stylegan"]):
        super().__init__(csv_name, is_training, transform_args)

        self.phase = phase

        self.toy = toy
        self.logger = logger
        
        self.csv_path = IMAGES_DIR / self.csv_name
        
        # Models to use
        self.models = models
        df = self.load_df(models)

        if self.toy and self.phase == 'train':
            df = df.sample(frac=0.01, random_state=42)
        
        if self.phase == 'train':
            self.img_paths = self.get_paths(df)
            self.labels = self.get_labels(df)
        else:
            # Evaluation mode
            self.img_paths, self.labels = self.get_expanded(df)

    def load_df(self, models):
        df = pd.read_csv(self.csv_path)
        return df[df['model'].isin(models)]

    def get_expanded(self, df):
        img_paths = []
        labels = []
        for i, row in df.iterrows():
            df_path = row['path']
            df_labels = str(row['labels'])
            for j in range(len(df_labels)):
                img_paths.append(df_path)
                labels.append(int(df_labels[j]))
        return img_paths, labels

    def get_paths(self, df):
        return df['path']

    def get_labels(self, df):
        return df['labels'].astype(str) 
 
    def get_image(self, index):
        if self.phase == 'train':
            # Get and transform the label
            labels = self.labels.iloc[index]

            # Select random label out of all labels
            label = int(random.choice(labels))
            
            # Get and transform the image
            img_path = self.img_paths.iloc[index]
        else:
            label = self.labels[index]
            img_path = self.img_paths[index]

        label = torch.FloatTensor([label])

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return img, label

    def __getitem__(self, index):
        return self.get_image(index)
