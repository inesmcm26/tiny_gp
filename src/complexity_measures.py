from scipy.spatial.distance import pdist
import numpy as np
import sympy as sp
from multiprocessing import Pool

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
    return corr / max_IODC

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

        # Get unique observations and their indices
        unique_obs, inverse_indices = np.unique(p_j, return_inverse=True)
        target_sums = np.zeros_like(unique_obs, dtype=np.float64)
        # Group by unique observations and average the target
        np.add.at(target_sums, inverse_indices, target)
        p_j = unique_obs
        target_j = target_sums / np.bincount(inverse_indices)

        print('UNIQUE OBS', p_j)
        print('AVG TARGET', target_j)
        
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
                first = 0
            else:
                first = (target_j[next_idx] - target_j[idx]) / (p_j[next_idx] - p_j[idx])

            if p_j[next_next_idx] == p_j[next_idx]:
                raise Exception('SOMETHING WRONG 2')
                second = 0
            else:
                second = (target_j[next_next_idx] - target_j[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])

            print('FIRST', first)
            print('SECOND', second)
            pc_j += abs(first - second)

        complexity += pc_j
    
    print('END INIT. MAX COMPLEXITY:', complexity)
    return complexity

def slope_based_complexity(max_complexity, best_ind, dataset):

    preds = [best_ind.compute_tree(obs) for obs in dataset]

    complexity = 0

    for j in range(dataset.shape[1]):
        
        # Values of feature j
        p_j = dataset[:, j].flatten()

        # Get unique observations and their indices
        unique_obs, inverse_indices = np.unique(p_j, return_inverse = True)
        preds_sums = np.zeros_like(unique_obs, dtype = np.float64)
        # Group by unique observations and average the target
        np.add.at(preds_sums, inverse_indices, preds)
        p_j = unique_obs
        preds_j = preds_sums / np.bincount(inverse_indices)

        print('UNIQUE OBS', p_j)
        print('AVG TARGET', preds_j)
        
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
                first = 0
            else:
                first = (preds_j[next_idx] - preds_j[idx]) / (p_j[next_idx] - p_j[idx])

            if p_j[next_next_idx] == p_j[next_idx]:
                raise Exception('SOMETHING WRONG 2')
                second = 0
            else:
                second = (preds_j[next_next_idx] - preds_j[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])

            print('FIRST', first)
            print('SECOND', second)
            pc_j += abs(first - second)

        complexity += pc_j

    # Normalize complexity
    return complexity / max_complexity

def mean_slope_based_complexity(max_complexity, p_js, q_js, population, dataset):
    complexities = []

    for ind in population:
        complexities.append(slope_based_complexity(max_complexity, p_js, q_js, ind, dataset))

    return np.mean(complexities)