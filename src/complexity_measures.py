from scipy.spatial.distance import pdist
import numpy as np
import sympy as sp
import pandas as pd
from ops import FUNCTIONS, MAPPING, add, sub, mul, div


def init_IODC(dataset, target):
    z = pdist(dataset)

    targets = [[target[i]] for i in range(len(target))]

    w = pdist(targets)

    max_IODC = abs(np.corrcoef(z, w)[0, 1])

    return z, max_IODC
    

def IODC(max_IODC, z, best_ind, dataset):
    """
    z = vector of pairwise distances between observations
    w = vector of pairwise distances of output values

    IODC = correlation(z, w)
    """

    preds = [[best_ind.compute_tree(obs)] for obs in dataset]

    w = pdist(preds)

    # Check for zero variance
    if np.var(z) != 0 and np.var(w) != 0:
        corr = abs(np.corrcoef(z, w)[0, 1])
    else:
        corr = 0.5 # TODO: should it be this way?

    # print('MEAN IODC', mean_iodc)
    
    # print('CORRELATION', corr)
    return 1 / (corr / max_IODC)

def mean_IODC(max_IODC, z, population, dataset):
    iodcs = []
    for ind in population:
        iodcs.append(IODC(max_IODC, z, ind, dataset))
    
    return np.mean(iodcs)

def polynomial_analysis(best_ind):

    # terminals = ' '.join(best_ind.terminals)

    # # [f'x{i}' for i in range(1, len(train_dataset[0]) + 1)]

    # # Define variables
    # x, y = sp.symbols(terminals)

    # Define the expression
    expression = best_ind.expression

    # print('EXPRESSION', expression)

    # Expand the expression
    expanded_expression = sp.expand(expression)

    # Print the expanded expression
    # print("EXPANDED EXPRESSION", expanded_expression)

    # Identify terms to represent unique interactions
    unique_interactions = expanded_expression.as_ordered_terms()

    # print('LEN UNIQUE INTERACTIONS', unique_interactions)

    return len(unique_interactions)

def mean_polynomial_analysis(population):
    p_analysis = []

    for ind in population:
        p_analysis.append(polynomial_analysis(ind))

    return p_analysis


def init_slope_based_complexity(dataset, target):
    complexity = 0

    for j in range(dataset.shape[1]):
        
        # Values of feature j
        p_j = dataset[:, j].flatten()

        # Calculate median prediction when there's more than one obs with same feature value
        df = pd.DataFrame({'Feature': p_j, 'Prediction': target})
        median_predictions = df.groupby('Feature')['Prediction'].median().reset_index()

        # Unique feature values
        p_j = median_predictions['Feature'].values
        # Unique feature values predictions
        preds_j = median_predictions['Prediction'].values
        
        # List of the ordered indexes of feature j
        q_j = np.argsort(p_j)

        pc_j = 0

        for i in range(len(q_j) - 2):
            idx = q_j[i]
            next_idx = q_j[i + 1]
            next_next_idx = q_j[i + 2]

            # TODO: 0 or continue when both are 0??

            if p_j[next_idx] == p_j[idx]:
                raise Exception('SOMETHING WRONG 1')
            else:
                first = (preds_j[next_idx] - preds_j[idx]) / (p_j[next_idx] - p_j[idx])

            if p_j[next_next_idx] == p_j[next_idx]:
                raise Exception('SOMETHING WRONG 2')
            else:
                second = (preds_j[next_next_idx] - preds_j[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])

            pc_j += abs(first - second)

        complexity += pc_j
    
    print('END INIT. MAX COMPLEXITY:', complexity)
    return complexity

def slope_based_complexity(ind, dataset):
    # Scale feature beforehand
    # Median of outputs of obs with same feature value

    preds = [ind.compute_tree(obs) for obs in dataset]

    # This is a list with the feature numbers
    used_feats = ind.used_feats()

    # print('ONLY USING FEATS:', used_feats, 'for ind:', ind.tree_lambda.expr)

    complexity = 0

    feats_complexity = {}

    # print('INDIVIDUAL:', ind.tree_lambda.expr)

    for j in used_feats:
        
        # Values of feature j
        p_j = dataset[:, j].flatten()

        # Calculate median prediction when there's more than one obs with same feature value
        df = pd.DataFrame({'Feature': p_j, 'Prediction': preds})
        median_predictions = df.groupby('Feature')['Prediction'].median().reset_index()

        # Unique feature values
        p_j = median_predictions['Feature'].values
        # Unique feature values predictions
        preds_j = median_predictions['Prediction'].values
     
        # List of the ordered indexes of feature j
        q_j = np.argsort(p_j)

        pc_j = 0

        for i in range(len(q_j) - 2):
            idx = q_j[i]
            next_idx = q_j[i + 1]
            next_next_idx = q_j[i + 2]

            if p_j[next_idx] == p_j[idx]:
                raise Exception('SOMETHING WRONG 1')
            else:
                first = (preds_j[next_idx] - preds_j[idx]) / (p_j[next_idx] - p_j[idx])

            if p_j[next_next_idx] == p_j[next_idx]:
                raise Exception('SOMETHING WRONG 2')
            else:
                second = (preds_j[next_next_idx] - preds_j[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])

            pc_j += abs(first - second)
    
        # Calculate the mean of the slopes difference across the i=1..n-2 sum
        mean_feat_complexity = (pc_j / (len(q_j) - 2)) if len(q_j) > 2 else 0

        feats_complexity[j] = mean_feat_complexity

        complexity = complexity + mean_feat_complexity
    
    # print('DICTIONARY:', feats_complexity)
    for feat in [int(term[1:]) - 1 for term in ind.terminals]:
        # print('FEATURE IN TERMINALS', feat)
        if feat not in feats_complexity.keys():
            # print('FEATURE NOT IN DICTIONARY', feat)
            feats_complexity[feat] = None

    ordered_feats_complexity = [value for key, value in sorted(feats_complexity.items())]

    # Returns a 
    return complexity, ordered_feats_complexity

def mean_slope_based_complexity(max_complexity, population, dataset):
    complexities = []

    for ind in population:
        complexities.append(slope_based_complexity(max_complexity, ind, dataset))

    return np.mean(complexities)