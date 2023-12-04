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

def plot_learning_curves(dataset_name, gp_method, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_train, mean_train_best = avg_results(results_path + gp_method + f'/{dataset_name}/', 'train')
    gens_test, mean_test_best = avg_results(results_path + gp_method + f'/{dataset_name}/', 'test')

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

def plot_mean_learning_curves(dataset_name, gp_method, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_train, mean_train_best = avg_results(results_path + gp_method + f'/{dataset_name}/' , 'mean_train')
    gens_test, mean_test_best = avg_results(results_path + gp_method + f'/{dataset_name}/', 'mean_test')

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

    if os.path.exists(base_path + f'{results_name}_run1.csv'):

        df = pd.read_csv(base_path + f'{results_name}_run1.csv', index_col = 0)

        for run_nr in range(2, 31):
            df_run = pd.read_csv(base_path + f'{results_name}_run{run_nr}.csv', index_col = 0)
            df = pd.concat([df, df_run], ignore_index = True)
    else:
       df = pd.read_csv(base_path + f'{results_name}.csv', index_col = 0) 
    
    return df.columns.values, df.median(axis = 0).to_numpy()

def plot_complexities(dataset_name, gp_method, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    gens_iodc, iodc_complexity = avg_results(results_path + gp_method + f'/{dataset_name}/', 'iodc_complexity')
    gens_slope, slope_complexity = avg_results(results_path + gp_method + f'/{dataset_name}/', 'slope_based_complexity')

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

def plot_learning_vs_complexity(dataset_name, gp_method, slope = False, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):

    fig = make_subplots(
        rows = 1,
        cols = 2,
        subplot_titles = [f'{dataset_name} Learning Curves', f'{dataset_name} Complexities'],
    )

    gens_train, mean_train_best = avg_results(results_path + gp_method + f'/{dataset_name}/', 'train')
    gens_test, mean_test_best = avg_results(results_path + gp_method + f'/{dataset_name}/', 'test')

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

    gens_iodc, iodc_complexity = avg_results(results_path + gp_method + f'/{dataset_name}/', 'iodc_complexity')    

    fig.add_trace(go.Scatter(x = gens_iodc,
                             y = iodc_complexity,
                             mode = 'lines',
                             name = f'{gp_method} IODC',
                             line = dict(color='red')),
                             row = 1, col = 2)
    if slope:
        gens_slope, slope_complexity = avg_results(results_path + gp_method + f'/{dataset_name}/', 'slope_based_complexity')
        
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


def plot_size_vs_fitness(dataset, results_path = '/home/ines/Documents/tese/tiny_gp/results_OpEq/'):

    train_fitness = pd.read_csv(results_path + dataset + f'/train_run1.csv', index_col = 0)
    mean_sizes = pd.read_csv(results_path + dataset + f'/mean_size_run1.csv', index_col = 0)

    for run_nr in [1, 2, 3, 4, 5, 6, 7, 8, 9, 11]:
        train_fitness_run = pd.read_csv(results_path + dataset +  f'/train_run{run_nr}.csv', index_col = 0)
        train_fitness = pd.concat([train_fitness, train_fitness_run], ignore_index = True)
        
        mean_sizes_run = pd.read_csv(results_path + dataset + f'/mean_size_run{run_nr}.csv', index_col = 0)
        mean_sizes = pd.concat([mean_sizes, mean_sizes_run], ignore_index = True)

    df = pd.DataFrame()
    df['Best Train Fitness'] = train_fitness.median(axis = 0)
    df['Avg Size'] = mean_sizes.median(axis = 0)

    fig = px.line(df, x="Avg Size", y="Best Train Fitness", title='Best Training Fitness Vs Solution Length')
    fig.show()

    
def calculate_mean_histogram(dataset, type, results_path = '/home/ines/Documents/tese/tiny_gp/results_OpEq/'):
    
    all_columns = []
        
    datasets = [results_path + dataset + f'/{type}_histogram_run{i}.csv' for i in [6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]]

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

    sorted_dataset.to_csv(results_path + dataset + f'/{type}_histogram.csv')
        

def plot_OpEq_distribution(dataset, type, results_path = '/home/ines/Documents/tese/tiny_gp/results_OpEq/'):
    
    mean_histogram_path = results_path + dataset + f'/{type}_histogram.csv'
    
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
        calculate_mean_histogram(dataset, type, results_path = results_path)
        
        plot_OpEq_distribution(dataset, type, results_path = results_path)


def plot_OpEq_all_distributions(results_path ='/home/ines/Documents/tese/tiny_gp/results_OpEq/'):
    
    datasets = os.listdir(results_path)
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
        
        pop_hist_path = results_path + dataset_name + '/population_histogram.csv'
        target_hist_path = results_path + dataset_name + '/target_histogram.csv'
        
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


def plot_complexity_distribution_over_gens(dataset, bound_max_bin = 50, bin_width = 0.05, max_bin_viz = 50, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):
    
    df = pd.read_csv(results_path + dataset + '/iodc_distribution_run6.csv', index_col = 0)


    # ---------------------------------- Checking max bin below the maximum pre-defined ---------------------------------- #
    # -------------------------------- Also checking how many individuals are left unseen -------------------------------- #
    for run_nr in [7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        df_run = pd.read_csv(results_path + dataset + f'/iodc_distribution_run{run_nr}.csv', index_col = 0)
        df = pd.concat([df, df_run], ignore_index = True)

    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)

    # Each column the values of all complexities in that generation over all runs
    concatenated_data = df.apply(lambda x: x.sum(), axis=0)

    # Find the highest ever complexity
    max_complexity = 0

    nr_inds_out_bound = 0
    maximum_ind_complexity = 0

    for col in range(len(concatenated_data.columns)):
        max_complexity_in_gen = max(concatenated_data.iloc[:, col])

        print('max complexity in gen:', max_complexity_in_gen)
        if max_complexity_in_gen < bound_max_bin:
            if max_complexity_in_gen > max_complexity:
                max_complexity = max_complexity_in_gen
        else:
            nr_inds_out = np.sum(concatenated_data.iloc[:, col] > bound_max_bin)
            nr_inds_out_bound += nr_inds_out
            print('NR INDS OUT OF BOUNDS', nr_inds_out)
            if max_complexity_in_gen > maximum_ind_complexity:
                maximum_ind_complexity = max_complexity_in_gen

    print('MAXIMUM COMPLEXITY EVER:', maximum_ind_complexity)
    print('ONLY COUNTING UNTIL MAX COMPLEXITY', max_complexity)
    print('NUMBER INDS OUT BOUND', nr_inds_out_bound)

    # ---------------------------------- Creating bin ranges ------------------------------- #
    # Create bins with width BIN_WIDTH
    custom_bins = np.arange(0, np.ceil(max_complexity) + 0.05, bin_width)

    # -------------------- Save dataset mean histogram over generations -------------------- #
    for run_nr in [6, 7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        df_run = pd.read_csv(results_path + dataset + f'/iodc_distribution_run{run_nr}.csv', index_col = 0)
    
        dist = []

        for col in df_run.columns:
            df_run[col] = df_run[col].apply(ast.literal_eval)

            # Create histogram with pre-defined bins
            counts, _ = np.histogram(df_run[col][0], bins = custom_bins)

            dist.append(list(counts))

        # Create a dataframe with shape generations x bins and the number of individuals per bin
        run_distribution = pd.DataFrame(dist, index = [idx for idx in range(1, len(dist) + 1)], columns = [f'bin_{col}' for col in range(1, len(dist[0]) + 1)])

        run_distribution.to_csv(results_path + dataset + f'/population_histogram_run{run_nr}.csv', )

    # Calculate mean bins frequency over the 30 runs
    calculate_mean_histogram(dataset = dataset, type = 'population', results_path = results_path)


    # ---------------------------- Plot the mean histogram over the generations ---------------------------- #
    mean_histogram_path = results_path + dataset + f'/population_histogram.csv'
    
    df = pd.read_csv(mean_histogram_path, index_col = 0)

    # Get the data
    generations = df.index.values
    bins = [int(bin.split('_')[-1]) for bin in df.columns.values]       
    
    # Limiting the vizualization window
    if max_bin_viz < bound_max_bin:
        bins = bins[:max_bin_viz]

    z = df.values.T

    fig = go.Figure(data=[go.Surface(z=z, x=generations, y=bins, coloraxis = "coloraxis")])
    fig.update_layout(scene=dict(
                    xaxis=dict(title='Generations'),
                    yaxis=dict(title='Bins'),
                    zaxis=dict(title='Population Distribution' if type == 'population' else 'Target Distribution', range=[0, 100]),
                    ),
                    title = dataset, autosize=False,
                    width=800, height=700,
                    margin=dict(l=40, r=40, b=40, t=60),
                    coloraxis = {'colorscale':'agsunset'})
    fig.show()


def plot_complexity_distribution_over_gens_old(dataset, bound_max_bin = None, bin_width = 0.01, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):

    df = pd.read_csv(results_path + '/StdGP/' + dataset + '/iodc_distribution_run16.csv', index_col = 0)

    for run_nr in range(17, 26):
        df_run = pd.read_csv(results_path + '/StdGP/' + dataset + f'/iodc_distribution_run{run_nr}.csv', index_col = 0)
        df = pd.concat([df, df_run], ignore_index = True)

    for col in df.columns:
        df[col] = df[col].apply(ast.literal_eval)
    
    # Each column the values of all complexities in that generation over all runs
    concatenated_data = df.apply(lambda x: x.sum(), axis=0)

    if bound_max_bin is None:

        # Find the highest ever complexity
        max_complexity = 0

        for col in concatenated_data.columns:
            if max(concatenated_data[col]) > max_complexity:
                max_complexity = max(concatenated_data[col])
    else:
        max_complexity = bound_max_bin

    # Create bins with width BIN_WIDTH
    custom_bins = np.arange(0, np.ceil(max_complexity) + 1, bin_width)

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


def plot_best_ind_bin_over_generations(dataset, bin_width = 0.01, results_path = '/home/ines/Documents/tese/tiny_gp/results_initial/'):
    
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


    best_ind_complexity = pd.read_csv(results_path + dataset + '/iodc_complexity_run6.csv', index_col = 0)

    for run_nr in [7, 8, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]:
        best_ind_complexity_run = pd.read_csv(results_path + dataset + f'/iodc_complexity_run{run_nr}.csv', index_col = 0)
        best_ind_complexity = pd.concat([best_ind_complexity, best_ind_complexity_run], ignore_index = True)
    

    def get_bin(x):
        return np.ceil(x / bin_width)
    
    best_bins_df = best_ind_complexity.apply(get_bin)

    fig = px.scatter()

    for row in range(best_bins_df.shape[0]):
        fig.add_trace(go.Scatter(x=np.arange(best_bins_df.shape[1] + 1), y = best_bins_df.iloc[row, :], mode = 'lines', name = f'Run {row}'))

    fig.update_layout(title='Bin of best individual', xaxis_title='Generations', yaxis_title='Bins',
                      xaxis=dict(autorange="reversed"), yaxis=dict(autorange="reversed"))

    fig.show()