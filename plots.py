import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    
def calculate_mean_histogram(dataset, type):
    
    all_columns = []
        
    datasets = [OPEQ_RESULTS_PATH + '/' + dataset + f'/{type}_histogram_run{i}.csv' for i in range(1, 31)]

    for ds in datasets:
        df = pd.read_csv(ds, index_col= 0)
        all_columns.extend(df.columns.values)
        
    all_columns = set(all_columns)

    all_dataframes = []
        
    for ds in datasets:
        df = pd.read_csv(ds, index_col=0)
        missing_columns = all_columns - set(df.columns.values)
        
        df[list(missing_columns)] = 0
        
        all_dataframes.append(df)

    all_data = pd.concat(all_dataframes, axis=1, join='outer')
    
    mean_df = pd.DataFrame()
    
    # Loop through each column set and calculate the mean
    for col in list(all_columns):
        # Select columns for the current set
        current_columns = all_data.filter(regex=f'{col}$')
        
        # Calculate the mean along columns for the current set
        mean_df[col] = current_columns.mean(axis=1)
    
    def sort_key(column_name):
        # Extract the numerical part of the column name
        num_part = int(column_name.split('_')[-1])
        return num_part

    # Sort columns using the custom sorting key
    sorted_columns = sorted(mean_df.columns, key=sort_key)

    sorted_dataset = mean_df[sorted_columns]

    sorted_dataset.to_csv(OPEQ_RESULTS_PATH + '/' + dataset + f'/{type}_histogram.csv')
        
        
    
def plot_OpEq_distribution(dataset, type):
    
    mean_histogram_path = OPEQ_RESULTS_PATH + '/' + dataset + f'/{type}_histogram.csv'
    
    if os.path.exists(mean_histogram_path):
        df = pd.read_csv(mean_histogram_path, index_col = 0)
        
        # Get the data
        generations = df.index.values
        bins = [int(bin.split('_')[-1]) for bin in df.columns.values]        
        z = df.values.T
        
        print(z.shape)

        fig = go.Figure(data=[go.Surface(z=z, x=generations, y=bins)])
        fig.update_layout(scene=dict(
                        xaxis=dict(title='Generations'),
                        yaxis=dict(title='Bins'),
                        zaxis=dict(title='Values', range=[0, 20]),
                        ),
                        title='Mt Bruno Elevation', autosize=False,
                        width=500, height=500,
                        margin=dict(l=65, r=50, b=65, t=90))
        fig.show()
    else:
        calculate_mean_histogram(dataset, type)
        
        plot_OpEq_distribution(dataset, type)


def plot_OpEq_all_distributions():
    
    datasets = os.listdir(OPEQ_RESULTS_PATH)
    # Create subplots
    cols = len(datasets)
    fig = make_subplots(rows = 2, cols=cols, subplot_titles=[f'{datasets[i]}' for i in range(len(datasets))],
                        specs=[[{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}],
                               [{'is_3d': True}, {'is_3d': True}, {'is_3d': True}, {'is_3d': True}]])

    for i, dataset_name in enumerate(datasets):
        
        pop_hist_path = OPEQ_RESULTS_PATH + '/' + dataset_name + '/population_histogram.csv'
        target_hist_path = OPEQ_RESULTS_PATH + '/' + dataset_name + '/target_histogram.csv'
        
        if not os.path.exists(pop_hist_path):
            calculate_mean_histogram(dataset_name, 'population')
        if not os.path.exists(target_hist_path):
            calculate_mean_histogram(dataset_name, 'target')
        
        # Read the DataFrame
        df_pop = pd.read_csv(pop_hist_path, index_col=0)
        df_target = pd.read_csv(target_hist_path, index_col=0)

        # Get the data
        generations = df_pop.index.values
        pop_bins = [int(bin.split('_')[-1]) for bin in df_pop.columns.values]
        pop_z = df_pop.values.T

        # Add surface to subplot
        fig.add_trace(go.Surface(z = pop_z, x = generations, y = pop_bins), row = 1, col = i + 1)

        # # Update layout for each subplot
        # fig.update_layout(scene = dict(
        #                 xaxis = dict(title = 'Generations'),
        #                 yaxis = dict(title = 'Bins'),
        #                 zaxis = dict(title = 'Population Frequency', range = [0, 20]),
        #             ),
        #             title = f'{dataset_name}',
        #             autosize = False,
        #             width = 1500, height = 500,
        #             margin = dict(l = 65, r = 50, b = 65, t = 50)
        #         )

        target_bins = [int(bin.split('_')[-1]) for bin in df_target.columns.values]
        target_z = df_target.values.T

        # Add surface to subplot
        fig.add_trace(go.Surface(z = target_z, x = generations, y = target_bins), row = 2, col = i + 1)

        # # Update layout for each subplot
        # fig.update_layout(scene = dict(
        #                 xaxis = dict(title = 'Generations'),
        #                 yaxis = dict(title = 'Bins'),
        #                 zaxis = dict(title = 'Target Frequency', range = [0, 20]),
        #             ),
        #             title = f'{dataset_name}',
        #             autosize = False,
        #             width = 1500, height = 500,
        #             margin = dict(l = 65, r = 50, b = 65, t = 50)
        #         )
        
    # Show the plot
    fig.show()