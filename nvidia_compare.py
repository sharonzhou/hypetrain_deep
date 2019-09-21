import pandas as pd
from pathlib import Path
from scipy import stats
import json

from constants import *

nvidia_csv = PROJECT_DIR / 'nvidia_realism_scores.csv'
dense_test_csv = IMAGES_DIR / 'dense_test.csv'

df_nvidia = pd.read_csv(nvidia_csv)
df_dense = pd.read_csv(dense_test_csv)

# Get intersection of filename col
df = pd.merge(df_nvidia, df_dense, how='inner', on=['filename'])

# Compute scores btw labels col and realism_score col
models = df['model'].unique()
metrics = { model: {} for model in models}
for model in models:
    metrics[model]['pearsonr'], metrics[model]['pearsonr_pval'] = stats.pearsonr(df.loc[df['model']==model, 'labels'], df.loc[df['model']==model, 'realism_score'])
    metrics[model]['spearmanr'], metrics[model]['spearmanr_pval'] = stats.spearmanr(df.loc[df['model']==model, 'labels'], df.loc[df['model']==model, 'realism_score'])
print(metrics)
print(json.dumps(metrics, sort_keys=True, indent=4))
import pdb;pdb.set_trace()



