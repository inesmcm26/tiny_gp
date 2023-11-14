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

    SAVE_PATH = f'/home/ines/Documents/tese/tiny_gp/results/StdGP/{ds_name}/'

    # Check if the directory exists
    if not os.path.exists(SAVE_PATH):
        # If the directory doesn't exist, create it
        os.makedirs(SAVE_PATH)

    train_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    test_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    train_best_ind = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    best_of_run_generations = pd.DataFrame(columns = ['Gen_Number'])
    mean_train_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    mean_test_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    iodc_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    p_analysis_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    mean_iodc_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    mean_p_analysis_results = pd.DataFrame(columns = [i for i in range(0, GENERATIONS + 1)])
    
    # Run for 30 times with each dataset partition
    for run_nr in tqdm(range(1, 31)):
        
        # Get correct data partition
        train_dataset, test_dataset, train_target, test_target = read_dataset(ds_name, run_nr)

        terminals = [f'x{i}' for i in range(1, len(train_dataset[0]) + 1)]

        # Run GP
        best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, mean_train_fit_list, mean_test_fit_list, \
            iodc_list, p_analysis_list, mean_iodc_list, mean_p_analysis_list = \
            evolve(train_dataset, test_dataset, train_target, test_target, terminals)

        # best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, = \
        #     evolve(train_dataset, test_dataset, train_target, test_target, terminals)
        
        train_fit = pd.DataFrame([best_train_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        test_fit = pd.DataFrame([best_test_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_ind = pd.DataFrame([best_ind_list], columns = [i for i in range(0, GENERATIONS + 1)])
        best_gen = pd.DataFrame([best_of_run_gen], columns = ['Gen_Number'])
        mean_train_fit = pd.DataFrame([mean_train_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_test_fit = pd.DataFrame([mean_test_fit_list], columns = [i for i in range(0, GENERATIONS + 1)])
        iodc = pd.DataFrame([iodc_list], columns = [i for i in range(0, GENERATIONS + 1)])
        p_analysis = pd.DataFrame([p_analysis_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_iodc = pd.DataFrame([mean_iodc_list], columns = [i for i in range(0, GENERATIONS + 1)])
        mean_p_analysis = pd.DataFrame([mean_p_analysis_list], columns = [i for i in range(0, GENERATIONS + 1)])
        
        train_results = pd.concat([train_results, train_fit], ignore_index = True)
        test_results = pd.concat([test_results, test_fit], ignore_index = True)
        train_best_ind = pd.concat([train_best_ind, best_ind], ignore_index = True)
        best_of_run_generations = pd.concat([best_of_run_generations, best_gen], ignore_index = True)
        mean_train_results = pd.concat([mean_train_results, mean_train_fit], ignore_index = True)
        mean_test_results = pd.concat([mean_test_results, mean_test_fit], ignore_index = True)
        iodc_results = pd.concat([iodc_results, iodc], ignore_index = True)
        p_analysis_results = pd.concat([p_analysis_results, p_analysis], ignore_index = True)
        mean_iodc_results = pd.concat([mean_iodc_results, mean_iodc], ignore_index = True)
        mean_p_analysis_results = pd.concat([mean_p_analysis_results, mean_p_analysis], ignore_index = True)

    train_results.to_csv(SAVE_PATH + 'train.csv')
    test_results.to_csv(SAVE_PATH + 'test.csv')
    train_best_ind.to_csv(SAVE_PATH + 'best_ind.csv')
    best_of_run_generations.to_csv(SAVE_PATH + 'best_gen.csv')
    mean_train_results.to_csv(SAVE_PATH + 'mean_train.csv')
    mean_test_results.to_csv(SAVE_PATH + 'mean_test.csv')
    iodc_results.to_csv(SAVE_PATH + 'iodc_complexity.csv')
    p_analysis_results.to_csv(SAVE_PATH + 'polynomial_complexity.csv')
    mean_iodc_results.to_csv(SAVE_PATH + 'mean_iodc_complexity.csv')
    mean_p_analysis_results.to_csv(SAVE_PATH + 'mean_polynomial_complexity.csv')

def run_StdGP_all_ds():
    DATA_PATH = '/home/ines/Documents/tese/tiny_gp/data'

    datasets = os.listdir(DATA_PATH)

    for dataset in datasets:
        print(f'-------------------------------- DATASET {dataset} --------------------------------')
        run_stdGP(dataset)


# run_stdGP('Concrete')

run_StdGP_all_ds()
