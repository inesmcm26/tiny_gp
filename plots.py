import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

RESULTS_PATH = '/home/ines/Documents/tese/tiny_gp/results1/'

def plot_learning_curves(dataset_name):
    """
    Plot the learning curves on train and test for all GP Methods on a given dataset
    """

    sns.set_theme()

    # Plotting multiple learning curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))  # Change the number of subplots and figure size as needed

    for gp_method in os.listdir(RESULTS_PATH):
        gens_train, train_results = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/train.csv')
        gens_test, test_results = avg_results(RESULTS_PATH + gp_method + f'/{dataset_name}/test.csv')

        # Plotting training curve
        sns.lineplot(x='Generation', y='Train Best Fitness',
                     data = pd.DataFrame({'Generation': gens_train, 'Train Best Fitness' : train_results}),
                     label=f'{gp_method} Train',
                     ax = axes[0])
        
        sns.lineplot(x='Generation', y='Test Best Fitness',
                     data = pd.DataFrame({'Generation': gens_test, 'Test Best Fitness' : test_results}),
                     label=f'{gp_method} Test',
                     ax = axes[1])

    axes[0].set_title('Train Learning Curves')
    axes[0].set_xlabel('Generation')
    axes[0].set_ylabel('Best Fitness')

    axes[1].set_title('Test Learning Curves')
    axes[1].set_xlabel('Generation')
    axes[1].set_ylabel('Best Fitness')

    axes[0].set_xticks(range(0, len(train_results) + 1, 10))
    axes[1].set_xticks(range(0, len(train_results) + 1, 10))

    plt.legend() # Show the legend
    plt.show()

def avg_results(path):
    df = pd.read_csv(path, index_col = 0)
    return df.columns.values, df.mean().to_numpy()
