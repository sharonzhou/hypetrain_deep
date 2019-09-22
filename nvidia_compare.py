import pandas as pd
from pathlib import Path
from scipy import stats
import json

from constants import *

nvidia_csv = PROJECT_DIR / 'nvidia_realism_scores.csv'
dense_test_csv = IMAGES_DIR / 'dense_test.csv'

df_nvidia = pd.read_csv(nvidia_csv)
df_dense = pd.read_csv(dense_test_csv)

#df_nvidia['realism_score_normalized'] = df_nvidia.apply(normalize[df_nvidia['realism_score'] > 0.5]
df_dense['labels_binarized'] = df_dense[df_dense['labels'] > 0.5]

# Get intersection of filename col
df = pd.merge(df_nvidia, df_dense, how='inner', on=['filename'])

# Compute scores btw labels col and realism_score col
models = df['model'].unique()
metrics = { model: {} for model in models}
for model in models:
    metrics[model]['pearsonr'], metrics[model]['pearsonr_pval'] = stats.pearsonr(df.loc[df['model']==model, 'labels'], df.loc[df['model']==model, 'realism_score'])
    metrics[model]['spearmanr'], metrics[model]['spearmanr_pval'] = stats.spearmanr(df.loc[df['model']==model, 'labels'], df.loc[df['model']==model, 'realism_score'])
    metrics[model]['auprc'] = skm.average_precision_score(df.loc[df['model']==model, 'labels_binarized'], df.loc[df['model']==model, 'realism_score_normalized'])
print(metrics)
print(json.dumps(metrics, sort_keys=True, indent=4))
import pdb;pdb.set_trace()



