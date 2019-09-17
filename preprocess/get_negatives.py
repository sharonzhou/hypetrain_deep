import os
import pandas as pd
import random
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path
import shutil

from constants import *
from download_data import get_size, download_image, get_args

# From StackOverflow: https://stackoverflow.com/questions/29545704/fast-haversine-approximation-python-pandas
def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


if __name__ == "__main__":

    args = get_args()

    # Initialize random seed
    random.seed(1)

    # Read in master csv of all positives
    master_positives_path = USGS_DIR / 'master_positives.csv'
    df_positives = pd.read_csv(master_positives_path)

    # US coordinates from Zhecheng
    N = 49.4
    S = 24.5
    E = -66.93
    W = -124.784

    label = 0 
    height = -1

    lats = df_positives['lat'].as_matrix()
    lons = df_positives['lon'].as_matrix()

    samples = []
    num_samples = 20000

    THRES_TRAIN = 10000
    THRES_VAL = 15000
    
    pbar = tqdm(total=num_samples)
    
    # Get samples equal to num_Samples
    while len(samples) < num_samples:
        # Randomly sample lat, lon coordinates
        lat = random.uniform(S, N)
        lon = random.uniform(W, E)

        dists_km = haversine_np(lons, lats, lon, lat) 

        # Get min dist in km
        km_per_side = .12
        if any(dists_km < km_per_side * math.sqrt(2)):
            # Reject as overlapping
            print(f'Reject: {min(dists_km)}')
        else:
            # Accept and increment num sampled
            #print(f'Accept: {min(dists_km)}')
            
            # Download image from gmaps
            size = get_size(lat)

            if len(samples) < THRES_TRAIN:
                split = 'train'
            elif len(samples) < THRES_VAL:
                split = 'validation'
            else:
                split = 'test'
            
            path_dir = USGS_DIR / split / 'negatives'
            os.makedirs(path_dir, exist_ok=True)
            filename = download_image(lat, lon, size, label, path_dir, args.api_key)
            if filename is not None:
                samples.append([filename, label, height, lat, lon, size, split])
                pbar.update(1)

    pbar.close()
    print(samples)

    COLUMNS = ['filename', 'label', 'height', 'lat', 'lon', 'size', 'split']
    df = pd.DataFrame(samples, columns=COLUMNS)

    path_master_negatives = USGS_DIR / 'master_negatives.csv'
    df.to_csv(str(path_master_negatives), index=False)

    splits = ['train', 'validation', 'test']
    for split in splits:
        path = USGS_DIR / f'{split}_negatives.csv'
        df_split = df[df['split'] == split]
        df_split.to_csv(str(path), index=False)
