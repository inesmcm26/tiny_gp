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

        print('DIMENSION J =', j)
        
        # Values of feature j
        p_j = dataset[:, j].flatten()

        print('FEATURE VALUES', p_j)
        
        # List of the ordered indexes of feature j
        q_j = np.argsort(p_j)

        print('SORTED INDEXES', q_j)

        pc_j = 0

        for i in range(len(q_j) - 2):
            idx = q_j[i]
            next_idx = q_j[i + 1]
            next_next_idx = q_j[i + 2]

            pc_j += abs(((target[next_idx] - target[idx]) / (p_j[next_idx] - p_j[idx])) - \
                ((target[next_next_idx] - target[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])))

        print('PC_J', pc_j)

        complexity += pc_j

    return complexity

def slope_based_complexity(max_complexity, best_ind, dataset):

    preds = [best_ind.compute_tree(obs) for obs in dataset]

    complexity = 0

    for j in range(dataset.shape[0]):
        
        # Values of feature j
        p_j = dataset[:, j].flatten()
        
        # List of the ordered indexes of feature j
        q_j = np.argsort(p_j)

        pc_j = 0

        for i in range(len(q_j) - 2):
            idx = q_j[i]
            next_idx = q_j[i + 1]
            next_next_idx = q_j[i + 2]

            pc_j += abs(((preds[next_idx] - preds[idx]) / (p_j[next_idx] - p_j[idx])) - \
                ((preds[next_next_idx] - preds[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])))

        complexity += pc_j

    # Normalize complexity
    return complexity / max_complexity


def mean_slope_based_complexity(max_complexity, population, dataset):

    complexities = {i : 0 for i in range(len(population))}

    all_preds = []
    
    # Save predictions of each individual
    for i in range(len(population)):
        all_preds.append([population[i].compute_tree(obs) for obs in dataset])
    
    # For each dimension
    for j in range(dataset.shape[1]):
        p_j = dataset[:, j].flatten()
            
        # List of the ordered indexes of feature j
        q_j = np.argsort(p_j)

        # Calculate the partial complexity of each individual in that dimension
        for i in range(len(population)):

            preds = all_preds[i]

            pc_j = 0

            # Go through all observations
            for i in range(len(q_j) - 2):
                idx = q_j[i]
                next_idx = q_j[i + 1]
                next_next_idx = q_j[i + 2]

                pc_j += abs(((preds[next_idx] - preds[idx]) / (p_j[next_idx] - p_j[idx])) - \
                    ((preds[next_next_idx] - preds[next_idx]) / (p_j[next_next_idx] - p_j[next_idx])))

            # Sum partial complexity to individual's complexity
            complexities[i] += pc_j

    complexs = np.array(list(complexities.values()))
    
    # Normalize by max complexity and return the mean
    return np.mean(complexs / max_complexity)

