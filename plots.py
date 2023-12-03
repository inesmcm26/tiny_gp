import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import numpy as np
import ast

RESULTS_PATH = '/home/ines/Documents/tese/tiny_gp/results_initial/'
OPEQ_RESULTS_PATH = '/home/ines/Documents/tese/tiny_gp/results_OpEq/'

BIN_WIDTH = 0.01


def plot_learning_curves(dataset_name, gp_method):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_train, mean_train_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'train')
    gens_test, mean_test_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'test')

    # Plotting training curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = gens_train,
                             y = mean_train_best,
                             mode = 'lines',
                             name = f'{gp_method} Train',
                             line = dict(color='blue')))
    
    fig.add_trace(go.Scatter(x = gens_test,
                             y = mean_test_best,
                             mode = 'lines',
                             name = f'{gp_method} Test',
                             line = dict(color='orange')))
    
    fig.update_layout(
        xaxis = dict(title = 'Generation'),
        yaxis = dict(title = 'RMSE'),
        xaxis_tickvals = list(range(0, len(mean_train_best) + 1, 100)),
        showlegend = True
    )

    fig.show()

def plot_mean_learning_curves(dataset_name, gp_method):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_train, mean_train_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/' , 'mean_train')
    gens_test, mean_test_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'mean_test')

    # Plotting training curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = gens_train,
                             y = mean_train_best,
                             mode = 'lines',
                             name = f'{gp_method} Mean Train',
                             line = dict(color='blue')))
    
    fig.add_trace(go.Scatter(x = gens_test,
                             y = mean_test_best,
                             mode = 'lines',
                             name = f'{gp_method} Mean Test',
                             line = dict(color='orange')))
    
    fig.update_layout(
        xaxis = dict(title = 'Generation'),
        yaxis = dict(title = 'RMSE', type = 'log'),
        xaxis_tickvals = list(range(0, len(mean_train_best) + 1, 100)),
        showlegend = True
    )

    fig.show()

def avg_results(base_path, results_name):

    df = pd.read_csv(base_path + f'{results_name}_run1.csv', index_col = 0)

    for run_nr in range(2, 31):
        df_run = pd.read_csv(base_path + f'{results_name}_run{run_nr}.csv', index_col = 0)
        df = pd.concat([df, df_run], ignore_index = True)
    
    return df.columns.values, df.median(axis = 0).to_numpy()

def plot_complexities(dataset_name, gp_method):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'iodc_complexity')
    gens_slope, slope_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'slope_based_complexity')

    # Plotting training curve
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x = gens_iodc,
                             y = iodc_complexity,
                             mode = 'lines',
                             name = f'{gp_method} IODC Complexity',
                             line = dict(color='blue')))
    
    fig.add_trace(go.Scatter(x = gens_slope,
                             y = slope_complexity,
                             mode = 'lines',
                             name = f'{gp_method} Slope Based Complexity',
                             line = dict(color='orange')))
    
    fig.update_layout(
        xaxis = dict(title = 'Generation'),
        yaxis = dict(title = 'Complexity'),
        xaxis_tickvals = list(range(0, len(gens_iodc) + 1, 100)),
        showlegend = True
    )

    fig.show()

def plot_learning_vs_complexity(dataset_name, gp_method):

    fig = make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = [f'{dataset_name} Learning Curves', f'{dataset_name} Complexities'],
    )

    gens_train, mean_train_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'train')
    gens_test, mean_test_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'test')

    fig.add_trace(go.Scatter(x = gens_train,
                             y = mean_train_best,
                             mode = 'lines',
                             name = f'{gp_method} Best Train',
                             line = dict(color='blue')),
                             row = 1, col = 1)
    
    fig.add_trace(go.Scatter(x = gens_test,
                             y = mean_test_best,
                             mode = 'lines',
                             name = f'{gp_method} Best Test',
                             line = dict(color='orange')),
                             row = 1, col = 1)

    gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'iodc_complexity')
    gens_slope, slope_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'slope_based_complexity')
    

    fig.add_trace(go.Scatter(x = gens_iodc,
                             y = iodc_complexity,
                             mode = 'lines',
                             name = f'{gp_method} IODC',
                             line = dict(color='red')),
                             row = 1, col = 2)
    
    fig.add_trace(go.Scatter(x = gens_slope,
                             y = slope_complexity,
                             mode = 'lines',
                             name = f'{gp_method} Slope',
                             line = dict(color='green')),
                             row = 1, col = 2)
    
    fig.update_layout(
        autosize=False,
        width=1200,
        height=500,
        margin=dict(l=40, r=20, b=70, t=70, pad=0),
        showlegend = True,
        legend=dict(x=0.5, y=-0.1, xanchor="center", orientation='h')
    )

    fig.show()
    
def plot_mean_learning_vs_complexity(dataset_name, gp_method):

    fig = make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = [f'{dataset_name} Mean Learning Curves', f'{dataset_name} Complexities'],
    )

    gens_train, mean_train_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'mean_train')
    gens_test, mean_test_best = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'mean_test')

    fig.add_trace(go.Scatter(x = gens_train,
                             y = mean_train_best,
                             mode = 'lines',
                             name = f'{gp_method} Mean Train',
                             line = dict(color='blue')),
                             row = 1, col = 1)
    
    fig.add_trace(go.Scatter(x = gens_test,
                             y = mean_test_best,
                             mode = 'lines',
                             name = f'{gp_method} Mean Test',
                             line = dict(color='orange')),
                             row = 1, col = 1)

    gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'mean_iodc_complexity')
    gens_slope, slope_complexity = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/', 'mean_slope_based_complexity')
    

    fig.add_trace(go.Scatter(x = gens_iodc,
                             y = iodc_complexity,
                             mode = 'lines',
                             name = f'{gp_method} Mean IODC',
                             line = dict(color='red')),
                             row = 1, col = 2)
    
    fig.add_trace(go.Scatter(x = gens_slope,
                             y = slope_complexity,
                             mode = 'lines',
                             name = f'{gp_method} Mean Slope',
                             line = dict(color='green')),
                             row = 1, col = 2)
    
    fig.update_yaxes(type='log', row=1, col=1)
    fig.update_yaxes(type='log', row=1, col=2)

    fig.update_layout(
        autosize=False,
        width=1200,
        height=500,
        margin=dict(l=40, r=20, b=70, t=70, pad=0),
        showlegend = True,
        legend=dict(x=0.5, y=-0.1, xanchor="center", orientation='h')
    )

    fig.show()

def complexity_overfitting_correlation(dataset_name):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    sns.set_theme()

    # Plotting multiple learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Change the number of subplots and figure size as needed

    gens_iodc, iodc_complexity = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/', 'iodc_complexity')
    gens_poly, poly_complexity = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/', 'polynomial_complexity')
    gens_overfitting, overfitting_results = avg_results(RESULTS_PATH + 'StdGP' + f'/{dataset_name}/', 'overfitting')
    
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


def plot_size_vs_fitness(dataset):

    train_fitness = pd.read_csv(OPEQ_RESULTS_PATH + dataset + f'/train_run1.csv', index_col = 0)
    mean_sizes = pd.read_csv(OPEQ_RESULTS_PATH + dataset + f'/mean_size_run1.csv', index_col = 0)

    for run_nr in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]:
        train_fitness_run = pd.read_csv(OPEQ_RESULTS_PATH + dataset +  f'/train_run{run_nr}.csv', index_col = 0)
        train_fitness = pd.concat([train_fitness, train_fitness_run], ignore_index = True)
        
        mean_sizes_run = pd.read_csv(OPEQ_RESULTS_PATH + dataset + f'/mean_size_run{run_nr}.csv', index_col = 0)
        mean_sizes = pd.concat([mean_sizes, mean_sizes_run], ignore_index = True)

    df = pd.DataFrame()
    df['Best Train Fitness'] = train_fitness.median(axis = 0)
    df['Avg Size'] = mean_sizes.median(axis = 0)

    fig = px.line(df, x="Avg Size", y="Best Train Fitness", title='Best Training Fitness Vs Solution Length')
    fig.show()

    
def calculate_mean_histogram(dataset, type):
    
    all_columns = []
        
    datasets = [OPEQ_RESULTS_PATH + dataset + f'/{type}_histogram_run{i}.csv' for i in range(1, 31)]

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

    sorted_dataset.to_csv(OPEQ_RESULTS_PATH + dataset + f'/{type}_histogram.csv')
        

def plot_OpEq_distribution(dataset, type):
    
    mean_histogram_path = OPEQ_RESULTS_PATH + dataset + f'/{type}_histogram.csv'
    
    if os.path.exists(mean_histogram_path):
        df = pd.read_csv(mean_histogram_path, index_col = 0)
        
        # Get the data
        generations = df.index.values
        bins = [int(bin.split('_')[-1]) for bin in df.columns.values]        
        z = df.values.T
        
        fig = go.Figure(data=[go.Surface(z=z, x=generations, y=bins, coloraxis = "coloraxis")])
        fig.update_layout(scene=dict(
                        xaxis=dict(title='Generations'),
                        yaxis=dict(title='Bins'),
                        zaxis=dict(title='Population Distribution' if type == 'population' else 'Target Distribution', range=[0, 20]),
                        ),
                        title = dataset, autosize=False,
                        width=800, height=700,
                        margin=dict(l=40, r=40, b=40, t=60),
                        coloraxis = {'colorscale':'agsunset'})
        fig.show()
    else:
        calculate_mean_histogram(dataset, type)
        
        plot_OpEq_distribution(dataset, type)


def plot_OpEq_all_distributions():
    
    datasets = os.listdir(OPEQ_RESULTS_PATH)
    # Create subplots
    cols = len(datasets)
    fig = make_subplots(
        rows = 2,
        cols = cols,
        subplot_titles = [f"{datasets[i]}" for i in range(len(datasets))],
        specs = [
            [{"type": 'scene'} for _ in range(cols)],
            [{"type": 'scene'} for _ in range(cols)],
        ],
        horizontal_spacing = 0.0000001,  # Adjust the width as needed
        vertical_spacing = 0.001, 
        row_titles = ['Population Distribution', 'Target Distribution']
    )

    # Define the default camera position for each subplot
    camera_params = dict(
        center=dict(x=0, y=0, z=0),  # Center of the scene
        eye=dict(x=1.5, y=1.5, z=1.5),      # Position of the camera
        up=dict(x=0, y=0, z=1)        # Up vector
    )

    for i, dataset_name in enumerate(datasets):
        
        pop_hist_path = OPEQ_RESULTS_PATH + dataset_name + '/population_histogram.csv'
        target_hist_path = OPEQ_RESULTS_PATH + dataset_name + '/target_histogram.csv'
        
        if not os.path.exists(pop_hist_path):
            calculate_mean_histogram(dataset_name, 'population')
        if not os.path.exists(target_hist_path):
            calculate_mean_histogram(dataset_name, 'target')
        
        # Read the DataFrame
        df_pop = pd.read_csv(pop_hist_path, index_col=0)
        df_target = pd.read_csv(target_hist_path, index_col=0)

        # Get the population data
        generations = df_pop.index.values
        pop_bins = [int(bin.split('_')[-1]) for bin in df_pop.columns.values]
        pop_z = df_pop.values.T
        
        # Get the target data
        target_bins = [int(bin.split('_')[-1]) for bin in df_target.columns.values]
        target_z = df_target.values.T

        # Add subplots
        fig.add_traces([go.Surface(z = pop_z, x = generations, y = pop_bins, coloraxis = "coloraxis"),
                        go.Surface(z = target_z, x = generations, y = target_bins, coloraxis = "coloraxis")],
                        rows = [1, 2], cols = [i+1, i+1])
        
        fig.update_layout(
            autosize=False,
            width=2000,
            height=1000,
            margin=dict( l=70, r=70, b=40, t=40, pad=0
            ),
            coloraxis = {'colorscale':'agsunset'}
        )
        
        # Add the z-labels
        fig.update_scenes(patch = dict(xaxis = dict(title_text = 'Generation'),
                                       yaxis = dict(title_text = 'Bin'),
                                       zaxis = dict(title_text = 'Population Frequency', range=[0, 40])),
                          camera = camera_params,
                          row = 1, col= i+1)
        

        fig.update_scenes(patch = dict(xaxis = dict(title_text = 'Generation'),
                                       yaxis = dict(title_text = 'Bin'),
                                       zaxis = dict(title_text = 'Target Frequency', range=[0, 40])),
                          camera = camera_params,
                          row = 2, col= i+1)
                    
    # Show the plot
    fig.show()



def plot_complexity_distribution_over_gens(dataset, bound_max_bin = None):

    df = pd.read_csv(RESULTS_PATH + '/StdGP/' + dataset + '/iodc_distributions_run1.csv', index_col = 0)

    for run_nr in range(2, 31):
        df_run = pd.read_csv(RESULTS_PATH + '/StdGP/' + dataset + f'/iodc_distributions_run{run_nr}.csv', index_col = 0)
        df = pd.concat([df, df_run], ignore_index = True)

    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)
    
    # Each column the values of all complexities in that generation over all runs
    concatenated_data = df.apply(lambda x: x.sum(), axis=0)

    if bound_max_bin is None:

        # Find the highest ever complexity
        max_bin = 0

        for col in concatenated_data.columns:
            if max(concatenated_data[col]) > max_bin:
                max_bin = max(concatenated_data[col])
    else:
        max_bin = bound_max_bin

    # Create bins with width BIN_WIDTH
    custom_bins = np.arange(0, np.ceil(max_bin), BIN_WIDTH)

    # Save distribution over generations
    dist = []

    for column in concatenated_data.columns:
        counts, _ = np.histogram(concatenated_data.loc[:, column], bins = custom_bins)

        dist.append(counts)

    # Save distribution on a dataframe and save the ranges of the bins as index
    histogram_over_gens = pd.DataFrame(dist).T
    histogram_over_gens.index = custom_bins[1:]

    # Save generations in X and bins in Y
    x, y = histogram_over_gens.columns.values, histogram_over_gens.index.values
    fig = go.Figure(data=[go.Surface(z=histogram_over_gens, y = y, x = x)])
    fig.update_layout(title='Mt Bruno Elevation', autosize=False,
                    width=500, height=500,
                    margin=dict(l=65, r=50, b=65, t=90),
                    scene=dict(
                        xaxis=dict(title='Generations'),
                        yaxis=dict(title='Bins'),
                        zaxis=dict(title='Frequency'),
                    ))
    fig.show()


def plot_best_ind_bin_over_generations(dataset):
    
    # df = pd.read_csv(RESULTS_PATH + '/StdGP/' + dataset + '/iodc_distributions.csv', index_col = 0)

    # for col in df.columns:
    #     df[col] = df[col].apply(ast.literal_eval)
    
    # # Each column the values of all complexities in that generation over all runs
    # concatenated_data = df.apply(lambda x: x.sum(), axis=0)

    # if bound_max_bin is None:

    #     # Find the highest ever complexity
    #     max_bin = 0

    #     for col in concatenated_data.columns:
    #         if max(concatenated_data[col]) > max_bin:
    #             max_bin = max(concatenated_data[col])
    # else:
    #     max_bin = bound_max_bin


    best_ind_complexity = pd.read_csv(RESULTS_PATH + '/StdGP/' + dataset + '/iodc_complexity_run1.csv', index_col = 0)

    for run_nr in range(2, 31):
        best_ind_complexity_run = pd.read_csv(RESULTS_PATH + '/StdGP/' + dataset + f'/iodc_complexity_run{run_nr}.csv', index_col = 0)
        best_ind_complexity = pd.concat([best_ind_complexity, best_ind_complexity_run], ignore_index = True)
    

    def get_bin(x):
        return np.ceil(x / BIN_WIDTH)
    
    best_bins_df = best_ind_complexity.apply(get_bin)

    fig = px.scatter()

    for row in range(best_bins_df.shape[0]):
        fig.add_trace(go.Scatter(x=np.arange(best_bins_df.shape[1] + 1), y = best_bins_df.iloc[row, :], mode = 'lines', name = f'Run {row}'))

    fig.update_layout(title='Bin of best individual', xaxis_title='Generations', yaxis_title='Bins',
                      xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))

    fig.show()