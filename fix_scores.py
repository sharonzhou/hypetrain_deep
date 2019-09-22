import csv
import pandas as pd

filenames = { 
        'final_scores.csv':6,
        'final_scores_all.csv':13, 
        'final_dense_scores.csv':9,
        'final_dense_scores_all.csv':11
        }

for filename, row_len in filenames.items():
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        new_rows = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row[:4] + ['pretrained'] + row[4:]
                continue
            if len(row) < row_len:
                # Add the pretrained param - which is after ts, trained_on, tested_on, tuned_on so index 4
                new_row = row[:4] + ["False"] + row[4:] 
                print(new_row)
            else:
                new_row = row
            new_rows.append(new_row)
    #new_filename = filename.split('.')[0]
    #new_filename = new_filename + '2.csv'
    new_filename = filename
    df = pd.DataFrame(new_rows, columns=header)
    df.to_csv(new_filename, index=False)
