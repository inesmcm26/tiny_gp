import pandas as pd
from random import uniform

for dataset in ['Bioavailability', 'Concrete', 'Toxicity', 'Istanbul']:

    for run_nr in range(1, 31):
        BASE_PATH = '/home/ines/Documents/tese/tiny_gp/data/' + dataset

        # Read train dataset
        train_df = pd.read_csv(BASE_PATH + f'/train_{run_nr}.csv', index_col = 0)

        artificial_dict = {}

        nr_obs_to_generate = train_df.shape[0]
        
        # For each feature
        for col_name in train_df.columns.values:
            if col_name != 'Target':
                # Generate random values for that feature
                col_rdm_values = [uniform(0, 1) for _ in range(nr_obs_to_generate)]

                # Add new feature values to dataframe
                artificial_dict[col_name] = col_rdm_values

        
        artificial_df = pd.DataFrame(artificial_dict)
        # Save artificially generated dataset
        artificial_df.to_csv(BASE_PATH + f'/val_{run_nr}.csv')