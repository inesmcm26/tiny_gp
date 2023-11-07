from scipy.spatial.distance import pdist
import numpy as np
import sympy as sp

def IODC(best_ind, dataset):
    """
    z = vector of pairwise distances between observations
    w = vector of pairwise distances of output values

    IODC = correlation(z, w)
    """

    z = pdist(dataset)

    preds = [[best_ind.compute_tree(obs)] for obs in dataset]

    w = pdist(preds)

    # Check for zero variance
    if np.var(z) != 0 and np.var(w) != 0:
        corr = np.corrcoef(z, w)[0, 1]
    else:
        corr = 0.5 # TODO: should it be this way?

    # print('MEAN IODC', mean_iodc)
    
    print('CORRELATION', corr)
    return corr


def polynomial_analysis(best_ind):

    # Define variables
    x, y = sp.symbols('x y')

    # Define the expression
    expression = x**2 * (x/y + y**3 / (x + y))

    # Expand the expression
    expanded_expression = sp.expand(expression)

    # Print the expanded expression
    print("Expanded Expression:", expanded_expression)

    # Identify terms to represent unique interactions
    unique_interactions = expanded_expression.as_ordered_terms()

    # Print the unique interactions
    print("Unique Interactions:")
    for term in unique_interactions:
        print(term)



