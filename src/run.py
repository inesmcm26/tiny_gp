import os
import pandas as pd
from tqdm import tqdm

from data import read_dataset
from tiny_gp import evolve
from configs import GENERATIONS

#####################################################
#                     StdGP Run                     #
#####################################################


def run_stdGP(ds_name):

    SAVE_PATH = f'/home/ines/Documents/tese/tiny_gp/results/{ds_name}/'

    # Check if the directory exists
    if not os.path.exists(SAVE_PATH):
        # If the directory doesn't exist, create it
        os.makedirs(SAVE_PATH)
    
    # Run for 30 times with each dataset partition
    for run_nr in tqdm(range(1, 31)): # TODO: CHANGE HERE!
        
        # Get correct data partition
        train_dataset, test_dataset, train_target, test_target = read_dataset(ds_name, run_nr)

        terminals = [f'x{i}' for i in range(1, len(train_dataset[0]) + 1)]

        # Run GP
        best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, \
            mean_train_fit_list, mean_test_fit_list, \
            iodc_list, mean_iodc_list, iodc_distribution_list, \
                 slope_list, mean_slope_list, slope_distribution_list, \
                     size_list, mean_size_list, size_distribution_list, \
                         no_list, mean_no_list, no_distribution_list, \
                            feats_list, mean_feats_list, feats_distribution_list = \
            evolve(train_dataset, test_dataset, train_target, test_target, terminals)
        
        train_fit = pd.DataFrame([best_train_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        test_fit = pd.DataFrame([best_test_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_ind = pd.DataFrame([best_ind_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_gen = pd.DataFrame([best_of_run_gen], columns = ['Gen_Number'])
        mean_train_fit = pd.DataFrame([mean_train_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_test_fit = pd.DataFrame([mean_test_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        iodc = pd.DataFrame([iodc_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_iodc = pd.DataFrame([mean_iodc_list], columns = [i for i in range(0, GENERATIONS + 1)])
        iodc_distribution = pd.DataFrame([iodc_distribution_list], columns = [i for i in range(0, GENERATIONS + 1)])
        slope = pd.DataFrame([slope_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_slope = pd.DataFrame([mean_slope_list], columns = [i for i in range(0, GENERATIONS + 1)])
        slope_distribution = pd.DataFrame([slope_distribution_list], columns = [i for i in range(0, GENERATIONS + 1)])
        size = pd.DataFrame([size_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_size = pd.DataFrame([mean_size_list], columns = [i for i in range(0, GENERATIONS + 1)])
        size_distribution = pd.DataFrame([size_distribution_list], columns = [i for i in range(0, GENERATIONS + 1)])
        no = pd.DataFrame([no_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_no = pd.DataFrame([mean_no_list], columns = [i for i in range(0, GENERATIONS + 1)])
        no_distribution = pd.DataFrame([no_distribution_list], columns = [i for i in range(0, GENERATIONS + 1)])
        feats = pd.DataFrame([feats_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_feats = pd.DataFrame([mean_feats_list], columns = [i for i in range(0, GENERATIONS + 1)])
        feats_distribution = pd.DataFrame([feats_distribution_list], columns = [i for i in range(0, GENERATIONS + 1)])


        train_fit.to_csv(SAVE_PATH + f'train_run{run_nr}.csv')
        test_fit.to_csv(SAVE_PATH + f'test_run{run_nr}.csv')
        best_ind.to_csv(SAVE_PATH + f'best_in_run{run_nr}.csv')
        best_gen.to_csv(SAVE_PATH + f'best_gen_run{run_nr}.csv')
        mean_train_fit.to_csv(SAVE_PATH + f'mean_train_run{run_nr}.csv')
        mean_test_fit.to_csv(SAVE_PATH + f'mean_test_run{run_nr}.csv')
        iodc.to_csv(SAVE_PATH + f'iodc_complexity_run{run_nr}.csv')
        mean_iodc.to_csv(SAVE_PATH + f'mean_iodc_complexity_run{run_nr}.csv')
        iodc_distribution.to_csv(SAVE_PATH + f'iodc_distribution_run{run_nr}.csv')
        slope.to_csv(SAVE_PATH + f'slope_complexity_run{run_nr}.csv')
        mean_slope.to_csv(SAVE_PATH + f'mean_slope_complexity_run{run_nr}.csv')
        slope_distribution.to_csv(SAVE_PATH + f'slope_distribution_run{run_nr}.csv')
        size.to_csv(SAVE_PATH + f'size_run{run_nr}.csv')
        mean_size.to_csv(SAVE_PATH + f'mean_size_run{run_nr}.csv')
        size_distribution.to_csv(SAVE_PATH + f'size_distribution_run{run_nr}.csv')
        no.to_csv(SAVE_PATH + f'no_run{run_nr}.csv')
        mean_no.to_csv(SAVE_PATH + f'mean_no_run{run_nr}.csv')
        no_distribution.to_csv(SAVE_PATH + f'no_distribution_run{run_nr}.csv')
        feats.to_csv(SAVE_PATH + f'num_feats_run{run_nr}.csv')
        mean_feats.to_csv(SAVE_PATH + f'num_mean_feats_run{run_nr}.csv')
        feats_distribution.to_csv(SAVE_PATH + f'num_feats_distribution_run{run_nr}.csv')

def run_StdGP_all_ds():
    DATA_PATH = '/home/ines/Documents/tese/tiny_gp/data'

    datasets = os.listdir(DATA_PATH)

    for dataset in datasets:
        print(f'-------------------------------- DATASET {dataset} --------------------------------')
        run_stdGP(dataset)

run_stdGP('Concrete')

# run_StdGP_all_ds()
