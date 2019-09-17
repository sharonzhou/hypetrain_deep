import argparse
import math
import requests
import pandas as pd
import urllib.request
import shutil
from pathlib import Path
import numpy as np
import os

from constants import *


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--api_key",
        type=str,
        #required=True,
        default='AIzaSyBEcQhmIz_sliJwqgHbQaBDUseGBxdQ6TA',
        help='Google Static Maps API key.'
    )

    args = parser.parse_args()

    return args


def get_size(lat):
    """
    Gets size of image in pixels from latitude of query.

    Params:
        lat (float)

    Returns:
        size (int)
    """
    m_per_side = 120
    m_per_pixel = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, GMAPS_ZOOM)
    size = math.ceil(m_per_side / m_per_pixel)
    return size

def download_image(lat, lon, size, label, path_dir, api_key):
    """
    Downloads and saves image from GMaps API in path_dir. Size is in pixels and calculated in get_size()

    Returns:
        path (str) to image or None if failed
    """
    url = "https://maps.googleapis.com/maps/api/staticmap"
    
    params =    {
                    'center': f'{lat},{lon}',
                    'zoom': GMAPS_ZOOM,
                    'size': f'{size}x{size}',
                    'maptype': 'satellite',
                    'key': api_key,
                }

    r = requests.get(url=url, params=params)
    if r.status_code == 200:
        filename = f'{lat}_{lon}_{size}px_{label}.png'
        path = str(path_dir / filename)
        with open(path, 'wb') as f:
            f.write(r.content)
        return filename
    else:
        return None

if __name__ == "__main__":

    args = get_args()

    label = 1

    df = pd.read_csv(USGS_DATA)
    # TODO: make csv of all positive patch locations (incl size) - this will be used to get negatives
    # Filter to use only samples that have confident loc/attr, with height data
    df = df[df['t_conf_loc'] == 3]
    df = df[df['t_conf_atr'] == 3]
    df = df[df['t_hh'].notnull()]
    print(len(df), 'examples')

    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    THRES_TRAIN = 10000
    THRES_VAL = 15000
    THRES_TEST = 20000
    num_processed = 0

    for i, row in df.iterrows():
        lat = row['ylat']
        lon = row['xlong']
        print(lat, lon)

        size = get_size(lat)

        if num_processed < THRES_TRAIN:
            split = 'train'
        elif num_processed < THRES_VAL:
            split = 'validation'
        elif num_processed < THRES_TEST:
            split = 'test'
        else:
            split = 'other_positive'
        
        path_dir = USGS_DIR / split / 'positives'
        os.makedirs(path_dir, exist_ok=True)
        filename = download_image(lat, lon, size, label, path_dir, args.api_key)
        if filename is not None:
            num_processed += 1
         
    print('Download completed')
