import csv
import pandas as pd

filenames = { 
        #'final_scores.csv':5,
        #'final_scores_all.csv':12, 
        'final_dense_scores.csv':8,
        'final_dense_scores_all.csv':10
        }

for filename, row_len in filenames.items():
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        new_rows = []
        for i, row in enumerate(reader):
            if i == 0:
                header = row
                continue
            if len(row) < row_len:
                # Add the tuned_on param - which is after ts, trained_on, tested_on, so index 3
                new_row = row[:4] + [""] + [""] + row[4:] 
                print(new_row)
            else:
                new_row = row
            new_rows.append(new_row)
    #new_filename = filename.split('.')[0]
    #new_filename = new_filename + '2.csv'
    new_filename = filename
    df = pd.DataFrame(new_rows, columns=header)
    df.to_csv(new_filename, index=False)
