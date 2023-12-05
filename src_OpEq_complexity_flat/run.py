import os
import pandas as pd
from tqdm import tqdm
import numpy as np

from data import read_dataset
from tiny_gp import evolve
from configs_OpEq_flat import GENERATIONS

#####################################################
#                     StdGP Run                     #
#####################################################


def run_stdGP(ds_name):

    SAVE_PATH = f'/home/ines/Documents/tese/tiny_gp/results_OpEq_complexity_flat/{ds_name}/'

    # Check if the directory exists
    if not os.path.exists(SAVE_PATH):
        # If the directory doesn't exist, create it
        os.makedirs(SAVE_PATH)
    
    # Run for 30 times with each dataset partition
    for run_nr in tqdm(range(2, 3)):
        
        # Get correct data partition
        train_dataset, test_dataset, train_target, test_target = read_dataset(ds_name, run_nr)

        terminals = [f'x{i}' for i in range(1, len(train_dataset[0]) + 1)]

        # Run GP
        best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, \
            iodc_list, target_histogram_list, population_histogram_list = \
            evolve(train_dataset, test_dataset, train_target, test_target, terminals)

        # Find the maximum length of the inner lists
        max_length = max(map(len, target_histogram_list))
        # Use NumPy to create an array with padded zeros and then convert it back to a list of lists
        padded_target_list = [list(np.pad(inner_list, (0, max_length - len(inner_list)), 'constant')) for inner_list in target_histogram_list]
        
        # Find the maximum length of the inner lists
        max_length = max(map(len, population_histogram_list))
        # Use NumPy to create an array with padded zeros and then convert it back to a list of lists
        padded_population_list = [list(np.pad(inner_list, (0, max_length - len(inner_list)), 'constant')) for inner_list in population_histogram_list]
        
        train_fit = pd.DataFrame([best_train_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        test_fit = pd.DataFrame([best_test_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_ind = pd.DataFrame([best_ind_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_gen = pd.DataFrame([best_of_run_gen], columns = ['Gen_Number'])
        iodc = pd.DataFrame([iodc_list], columns = [i for i in range(0, GENERATIONS + 1)])
        target_histogram = pd.DataFrame(padded_target_list, columns = [f'bin_{i}' for i in range(1, len(padded_target_list[0]) + 1)])
        population_histogram = pd.DataFrame(padded_population_list, columns = [f'bin_{i}' for i in range(1, len(padded_population_list[0]) + 1)])

        train_fit.to_csv(SAVE_PATH + f'train_run{run_nr}.csv')
        test_fit.to_csv(SAVE_PATH + f'test_run{run_nr}.csv')
        best_ind.to_csv(SAVE_PATH + f'best_in_run{run_nr}.csv')
        best_gen.to_csv(SAVE_PATH + f'best_gen_run{run_nr}.csv')
        iodc.to_csv(SAVE_PATH + f'iodc_complexity_run{run_nr}.csv')
        target_histogram.to_csv(SAVE_PATH + f'target_histogram_run{run_nr}.csv')
        population_histogram.to_csv(SAVE_PATH + f'population_histogram_run{run_nr}.csv')
    
def run_StdGP_all_ds():
    DATA_PATH = '/home/ines/Documents/tese/tiny_gp/data'

    datasets = os.listdir(DATA_PATH)

    for dataset in datasets:
        print(f'-------------------------------- DATASET {dataset} --------------------------------')
        run_stdGP(dataset)


# run_stdGP('Concrete')

# run_StdGP_all_ds()
run_stdGP('Bioavailability')