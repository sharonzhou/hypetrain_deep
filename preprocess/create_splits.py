import argparse
import math
import requests
import pandas as pd
import os
import shutil

from constants import *
import urllib.request

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api_key",
        type=str,
        required=True,
        help='Google Static Maps API key.'
    )

    parser.add_argument(
        "--use_formula",
        action="store_true",
        help='Use formula to compute meters per pixel.'
    )

    args = parser.parse_args()

    return args

def save_csv(paths, split):
    COLUMNS = ['path', 'label', 'height', 'lat', 'lon', 'size']
    df = pd.DataFrame(paths, columns=COLUMNS) 
    df.to_csv(str(USGS_DIR / f'{split}.csv'), index=False)

if __name__ == "__main__":

    #args = get_args()

    df = pd.read_csv(USGS_DATA)
    # TODO: make csv of all positive patch locations (incl size) - this will be used to get negatives
    # Filter to use only samples that have confident loc/attr, with height data
    df = df[df['t_conf_loc'] == 3]
    df = df[df['t_conf_atr'] == 3]
    df = df[df['t_hh'].notnull()]
    print(len(df), 'examples')

    zoom = 20
    m_per_side = 120

    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    THRES_TRAIN = 10000
    THRES_VAL = 15000
    THRES_TEST = 20000

    num_processed = 0

    train_paths = []
    valid_paths = []
    test_paths = []
    other_paths = []
    unprocessed = [] # Files that failed to download
  
    for i, row in df.iterrows():
        lat = row['ylat']
        lon = row['xlong']
        height = row['t_hh']
        label = 1
        
        m_per_pixel = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, zoom)
        size = math.ceil(m_per_side / m_per_pixel)
        
        file_name = f'{lat}_{lon}_{size}px'
        
        row = [file_name, label, height, lat, lon, size]
        
        original_path = USGS_DIR / 'other_positive' / f'{file_name}.png'
        if os.path.exists(original_path):
            
            if num_processed < THRES_TRAIN:
                split = 'train'
                train_paths.append(row)
            elif num_processed < THRES_VAL:
                split = 'validation'
                valid_paths.append(row)
            elif num_processed < THRES_TEST:
                split = 'test'
                test_paths.append(row)
            else:
                split = 'other_positive'
                other_paths.append(row)

            file_path = USGS_DIR / split / f'{file_name}.png'
            shutil.move(original_path, file_path)
            num_processed += 1
        else:

            unprocessed.append(row)
        
    save_csv(train_paths, 'train')
    save_csv(valid_paths, 'validation')
    save_csv(test_paths, 'test')
    save_csv(other_paths, 'other_positive')
    save_csv(unprocessed, 'unprocessed')
       
    print('Completed move')
 
