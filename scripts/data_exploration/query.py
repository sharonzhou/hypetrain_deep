import argparse
import math
import requests
import pandas as pd

from constants import *


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


if __name__ == "__main__":

    args = get_args()

    df = pd.read_csv(USGS_DATA)

    url = "https://maps.googleapis.com/maps/api/staticmap"

    zoom = 20
    m_per_side = 30

    df = df.sample(frac=1).reset_index(drop=True)

    for i, row in df.iterrows():
        lat = row['ylat']
        lon = row['xlong']
        print(lat, lon)

        if args.use_formula:

            # TODO: Check if this works and if it is necessary to use.
            m_per_pixel = 156543.03392 * math.cos(lat * math.pi / 180) / math.pow(2, zoom)
            size = math.ceil(m_per_side / m_per_pixel)

        else:
            size = 300

        params =    {
                        'center': f'{lat},{lon}',
                        'zoom': zoom,
                        'size': f'{size}x{size}',
                        'maptype': 'satellite',
                        'key': args.api_key,
                    }
        
        r = requests.get(url=url, params=params)
        if r.status_code == 200:
            print(r.url)
        else:
            print('Error', r.status_code)
        
        if i > 20:
            break
