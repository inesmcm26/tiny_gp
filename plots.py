import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATH = '/home/ines/Documents/tese/tiny_gp/results/'
OPEQ_RESULTS_PATH = '/home/ines/Documents/tese/tiny_gp/results_OpEq'

def plot_learning_curves(dataset_name, gp_method):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    sns.set_theme()

    gens_train, train_results = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/train.csv')
    gens_test, test_results = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/test.csv')

    # Plotting training curve
    sns.lineplot(x='Generation', y='Train Best Fitness',
                    data = pd.DataFrame({'Generation': gens_train, 'Train Best Fitness' : train_results}),
                    label=f'{gp_method} Train')
    
    sns.lineplot(x='Generation', y='Test Best Fitness',
                    data = pd.DataFrame({'Generation': gens_test, 'Test Best Fitness' : test_results}),
                    label=f'{gp_method} Test')

    plt.xlabel('Generation')
    plt.ylabel('Best Fitness')


    plt.xticks(range(0, len(train_results) + 1, 10))
    plt.xticks(range(0, len(train_results) + 1, 10))

    plt.legend() # Show the legend
    plt.show()

def avg_results(path):
    df = pd.read_csv(path, index_col = 0)
    return df.columns.values, df.mean().to_numpy()

def plot_complexities(dataset_name):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    sns.set_theme()

    # Plotting multiple learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Change the number of subplots and figure size as needed

    for gp_method in os.listdir(RESULTS_PATH):
        gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/iodc_complexity.csv')
        gens_poly, poly_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/polynomial_complexity.csv')

        # Plotting training curve
        sns.lineplot(x='Generation', y='IODC Complexity',
                     data = pd.DataFrame({'Generation': gens_iodc, 'IODC Complexity' : iodc_complexity}),
                     label=f'{gp_method} IODC',
                     ax = axes[0])
        
        sns.lineplot(x='Generation', y='Polynomial Complexity',
                     data = pd.DataFrame({'Generation': gens_poly, 'Polynomial Complexity' : poly_complexity}),
                     label=f'{gp_method} Polynomial',
                     ax = axes[1])

    axes[0].set_title('IODC Complexity')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Complexity')

    axes[1].set_title('Polynomial Complexity')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Complexity')

    axes[0].set_xticks(range(0, len(iodc_complexity) + 1, 10))
    axes[1].set_xticks(range(0, len(poly_complexity) + 1, 10))

    plt.legend() # Show the legend
    plt.show()

def complexity_overfitting_correlation(dataset_name):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    sns.set_theme()

    # Plotting multiple learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Change the number of subplots and figure size as needed

    gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/iodc_complexity.csv')
    gens_poly, poly_complexity = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/polynomial_complexity.csv')
    gens_overfitting, overfitting_results = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/overfitting.csv')
    
    # Plotting training curve
    sns.scatterplot(x = 'IODC Complexity', y = 'Overfitting',
                    data = pd.DataFrame({'Generation': gens_iodc, 'IODC Complexity' : iodc_complexity, 'Overfitting': overfitting_results}),
                    ax = axes[0])
    
    sns.scatterplot(x = 'Polynomial Complexity', y = 'Overfitting',
                    data = pd.DataFrame({'Generation': gens_poly, 'Polynomial Complexity' : poly_complexity, 'Overfitting': overfitting_results}),
                    ax = axes[1])

    axes[0].set_title('IODC-Overfitting Correlation')
    axes[0].set_xlabel('IODC Complexity')
    axes[0].set_ylabel('Overfitting')

    axes[1].set_title('Polynomial-Overfitting Correlation')
    axes[1].set_xlabel('Polynomial Complexity')
    axes[1].set_ylabel('Overfitting')

    axes[0].set_xticks(range(0, len(iodc_complexity) + 1, 10))
    axes[1].set_xticks(range(0, len(poly_complexity) + 1, 10))

    plt.legend() # Show the legend
    plt.show()
    
    
    
def plot_OpEq_distribution(dataset, type):
    
    all_columns = []
    
    datasets = [OPEQ_RESULTS_PATH + '/' + dataset + f'/{type}_histogram_run{i}.csv' for i in range(1, 31)]
    
    for ds in datasets:
        df = pd.read_csv(ds)
        all_columns = all_columns.extend(df.columns.values)
        
    all_columns = set(all_columns)
    
    all_dataframes = []
        
    for ds in datasets:
        df = pd.read_csv(ds)
        missing_columns = all_columns - set(df.columns.values)
        
        df[list(missing_columns)] = 0
        
        all_dataframes.append(df)
        
    all_data = pd.concat(all_dataframes, axis=1, join='outer')
    
    print(all_data)
    print(all_data.columns)
    
    # mean_dataset = merged_dataset.mean(axis=1)
    