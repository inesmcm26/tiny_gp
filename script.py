import os
import pandas as pd

DATA_FOLDER = '/home/ines/Documents/tese/tiny_gp/Bioavalability/'
NR_VARS = 241

VARS = [f'x{i}' for i in range(1, NR_VARS + 1)] + ['Target']

print(VARS)


for file in os.listdir(DATA_FOLDER):
    df = pd.DataFrame(columns = VARS)

    file_nr = file.split('_')[1]

    print('FILE', file)
    print('FILE NR', file_nr)

    # Read file
    with open(DATA_FOLDER + file) as f:

        for line in f.readlines():

            print(line + '\n--------\n')

            # Do something with the line (e.g., print it)
            values = line.split(' ')[:-1]

            print(len(values))

            # Create a new DataFrame for the new row
            new_row_df = pd.DataFrame([values], columns = VARS)

            # Concatenate the original DataFrame with the new row
            df = pd.concat([df, new_row_df], ignore_index=True)

    if 'TEST' in file:
        file_name = DATA_FOLDER + 'test_' + file_nr + '.csv'
    else:
        file_name = DATA_FOLDER + 'train_' + file_nr + '.csv'
    
    df.to_csv(file_name)
