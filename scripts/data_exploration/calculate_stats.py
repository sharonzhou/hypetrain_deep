import pandas as pd
import numpy as np
from collections import Counter

from constants import *


# Filter data to only include confident locations and attrs, and incl height
df = pd.read_csv(USGS_DATA)
df = df[df['t_conf_loc'] == 3]
print(len(df), 'examples filtered for location confidence')
df = df[df['t_conf_atr'] == 3]
print(len(df), 'examples filtered again for atr confidence')
df = df[df['t_hh'].notnull()]
print(len(df), 'examples filtered yet again for height')
df = df[(df['ylat'] > 24.5) & (df['ylat'] < 49.4)]
df = df[(df['xlong'] < -66.93) & (df['xlong'] > -124.784)]
print(len(df), 'examples filtered for US coordinates only')

# Calculate info on heights (range min/max, mean, median, std)
max_height = df['t_hh'].max()
min_height = df['t_hh'].min()
avg_height = df['t_hh'].mean()
std_height = df['t_hh'].std()
median_height = df['t_hh'].median()

print(f'Height. Max: {max_height}. Min: {min_height}. Avg: {avg_height}. Std: {std_height}. Median: {median_height}')

rows_with_max_height = df[['ylat', 'xlong', 't_hh']][df['t_hh'] == max_height]
print(f'Max height turbines: {rows_with_max_height}')







