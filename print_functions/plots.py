import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from parser import tokenize, parse
from data import read_dataset

MAPPTING_METHODS = {
    'StdGP': 'results_elitism/results',
    'Double Tournament (Prob=1)': 'results_elitism/results_nested',
    'Inverted Double Tournament (Prob=1)': 'results_elitism/results_inverted_tournament',
    'Double Tournament (Prob=0.7)': 'results_elitism/results_nested_prob_0.7',
    'Inverted Double Tournament (Prob=0.7)': 'results_elitism/results_inverted_nested_prob_0.7',
    'Double Tournament (Prob=0.5)': 'results_elitism/results_nested_prob_0.5',
    'Inverted Double Tournament (Prob=0.5)': 'results_elitism/results_inverted_nested_prob_0.5',
    'MMOTS': 'results_elitism/mmots',
    'Subsampled': 'results_elitism/results_nested_subsampled',
    'Oversampled': 'results_elitism/results_nested_oversampled',
    'DT_2_2': 'results_elitism/results_nested_2_2',
    'DT_2_4': 'results_elitism/results_nested_2_4',
    'DT_4_2': 'results_elitism/results_nested',
    'DT_4_4': 'results_elitism/results_nested_4_4',
    'DT_10_2': 'results_elitism/results_nested_10_2',
    'DT_10_4': 'results_elitism/results_nested_10_4',
    'Double Tournament Complexity Limit': 'results_elitism/results_nested_limit',
}

GENS = [10, 250, 500]
def plot_curvatures(dataset, methods, run_nr, base_path, saving_path = None):
    X, _, Y, _ = read_dataset(dataset, run_nr)

    # Read dataset
    df = pd.read_csv(base_path + f'/data/{dataset}/train_{run_nr}.csv', index_col=0)
    terminals = df.drop('Target', axis=1).columns

    fig = make_subplots(rows=len(terminals), cols=len(GENS),
                        subplot_titles=[f'Generation {gen}' for gen in GENS])

    colors = ['blue', 'orange', 'red', 'green']

    not_used = {idx: [] for idx in range(len(GENS))}

    for method_idx, m in enumerate(methods):
        method_path = base_path + f'/{MAPPTING_METHODS[m]}/{dataset}/'

        # Read and plot function
        best_of_run = pd.read_csv(method_path + f'best_in_run{run_nr}.csv', index_col = 0)
        best_of_runs = best_of_run.iloc[0, GENS]
        print(best_of_runs)

        # Define symbols for the lambda functions
        for col, best_of_run in enumerate(best_of_runs):
            tokens = tokenize(best_of_runs[col])
            gptree = parse(tokens, terminals)
            gptree.create_lambda_function()

            for row, terminal in enumerate(terminals):
                if '('+ terminal + ')' in best_of_run:

                    feat_values = X[:, row].flatten()
                    preds = [gptree.compute_tree(obs) for obs in X]


                    # Calculate median prediction when there's more than one obs with same feature value
                    feats_df = pd.DataFrame({'Feature': feat_values, 'Prediction': preds})
                    median_predictions = feats_df.groupby('Feature')['Prediction'].median().reset_index()

                    # Unique feature values
                    p_j = median_predictions['Feature'].values
                    # Unique feature values predictions
                    preds_j = median_predictions['Prediction'].values

                    fig.add_trace(go.Scatter(x=p_j, y=preds_j, mode='lines',
                                            line=dict(color=colors[method_idx], width = 2)),
                                            row=row+1, col=col+1)
                    
                    if method_idx == 0:
                        fig.update_xaxes(title_text=terminal, row=row+1, col=col+1)
                        # fig.update_yaxes(title_text='Prediction', row=row+1, col=col+1)
                else:
                    fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                            line=dict(color=colors[method_idx], width = 2),
                                            text='Feature not used'),
                                            row=row+1, col=col+1)
                    

                    not_used[col].append(row)

        for col in not_used:
            for row in not_used[col]:
                count = not_used[col].count(row)
                if count == len(methods):
                    # Add annotation for feature not used
                    fig.add_annotation(
                        xref="x domain",
                        yref="y domain",
                        x=0.5, y=0.5,
                        text="Feature not used",
                        showarrow=False,
                        font=dict(size=16),
                        row=row + 1, col=col + 1
                    )
                    
        
        fig.update_layout(
            autosize=False,
            width=1000,
            height=2500,
            margin=dict(l=40, r=20, b=70, t=70, pad=0),
            showlegend = False,
            legend=dict(yanchor="bottom", xanchor="center", x=0.5, orientation='h'),
        )

    fig.show()

    if saving_path:
        fig.write_image(saving_path)

