import os
from tqdm import tqdm
import pandas as pd
from data import read_dataset
from complexity_measures_limit import init_IODC, init_slope_based_complexity

def get_max_complexities(ds_name):

    SAVE_PATH = f'/home/ines/Documents/tese/tiny_gp/results/{ds_name}/'

    # Check if the directory exists
    if not os.path.exists(SAVE_PATH):
        # If the directory doesn't exist, create it
        os.makedirs(SAVE_PATH)

    complexities = pd.DataFrame({'Max IODC': [], 'Max Slope': []})
    
    # Run for 30 times with each dataset partition
    for run_nr in tqdm(range(1, 31)): # TODO: CHANGE HERE!
        
        # Get correct data partition
        train_dataset, test_dataset, train_target, test_target = read_dataset(ds_name, run_nr)

        z, max_IODC = init_IODC(train_dataset, train_target)

        max_slope_complexity = init_slope_based_complexity(train_dataset, train_target)

        run_complexities = pd.DataFrame({'Max IODC': [max_IODC], 'Max Slope': [max_slope_complexity]})

        complexities = pd.concat([complexities, run_complexities], axis = 0)

    
    complexities.to_csv(SAVE_PATH + 'max_complexities.csv')


get_max_complexities('Istanbul')