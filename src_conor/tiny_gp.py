from random import randint, random, choice
from copy import deepcopy
import numpy as np
import time

from configs_variance import *
from gptree import GPTree
from complexity_measures import slope_based_complexity
                   
def init_population(terminals):
    """
    Ramped half-and-half initialization
    """

    # Number of individuals of each depth and initialized with each method
    inds_per_depth = int((POP_SIZE / (MAX_INITIAL_DEPTH)) / 2)

    pop = []
    pop_str = []

    for max_depth in range(MIN_DEPTH, MAX_INITIAL_DEPTH + 1):

        print('MAX DEPTH', max_depth)

        # Grow
        for _ in range(inds_per_depth):

            for _ in range(20): 
                ind = GPTree(terminals = terminals)
                ind.random_tree(grow = True, max_depth = max_depth)
                ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

                if ind.tree2_string() not in pop_str:
                    break

            pop.append(ind) 
            pop_str.append(ind.tree2_string())
        
        # Full
        for _ in range(inds_per_depth):

            for _ in range(20):
                ind = GPTree(terminals = terminals)
                ind.random_tree(grow = False, max_depth = max_depth)  
                ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

                if ind.tree2_string() not in pop_str:
                    break


            pop.append(ind)
            pop_str.append(ind.tree2_string())

    # Edge case
    while len(pop) != POP_SIZE:
        for _ in range(20):
            # Generate random tree with grow method at higher level to fill population
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = True, max_depth = MAX_INITIAL_DEPTH)
            ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

            if ind.tree2_string() not in pop_str:
                break

        pop.append(ind)
        pop_str.append(ind.tree2_string())

    return pop

def fitness(individual, dataset, target):

    # Calculate predictions
    preds = [individual.compute_tree(obs) for obs in dataset]
    
    if FITNESS == 'RMSE':
        return np.sqrt(np.mean((np.array(preds) - np.array(target)) ** 2))
    
    elif FITNESS == 'MAE':
        return np.mean(abs(np.array(preds) - np.array(target)))

def compute_variance(individual, dataset):

    preds = [individual.compute_tree(obs) for obs in dataset]

    return np.var(preds)


def tournament(population, fitnesses, variances):
    """
    Tournament selection: modified tournament selection scheme
    """
    # Select random individuals to compete
    tournament_idxs = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]
    
    # Get their variance values
    variance_values = [variances[ind_idx] for ind_idx in tournament_idxs]
    # Get their fitness values
    tournament_fitnesses = [fitnesses[ind_idx] for ind_idx in tournament_idxs]

    # A dominates over B
    if (tournament_fitnesses[0] <= tournament_fitnesses[1] and variance_values[0] < variance_values[1]) or \
        (tournament_fitnesses[0] < tournament_fitnesses[1] and variance_values[0] <= variance_values[1]):
        return deepcopy(population[tournament_idxs[0]])
    # B dominates over A
    elif (tournament_fitnesses[1] <= tournament_fitnesses[0] and variance_values[1] < variance_values[0]) or \
        (tournament_fitnesses[1] < tournament_fitnesses[0] and variance_values[1] <= variance_values[0]):
        return deepcopy(population[tournament_idxs[1]])
    # No dominance. Rectilinear distance of A is smaller than euclidean distance of B
    elif abs(tournament_fitnesses[0] + variance_values[0]) < np.sqrt(tournament_fitnesses[1]**2 + variance_values[1]**2):
        return deepcopy(population[tournament_idxs[0]])
    # No dominance. Rectilinear distance of B is smaller than euclidean distance of A
    elif abs(tournament_fitnesses[1] + variance_values[1]) < np.sqrt(tournament_fitnesses[0]**2 + variance_values[0]**2):
        return deepcopy(population[tournament_idxs[1]])
    # Return individual with smaller variance
    elif variance_values[0] < variance_values[1]:
        return deepcopy(population[tournament_idxs[0]])
    elif variance_values[1] < variance_values[0]:
        return deepcopy(population[tournament_idxs[1]])
    # Return individual with smaller size
    else:
        sizes = [population[tournament_idxs[0]].size(), population[tournament_idxs[1]].size()]
        return deepcopy(population[tournament_idxs[sizes.index(min(sizes))]])


def evolve(train_dataset, test_dataset, train_target, test_target, terminals):

    # print('TRAIN DATASET')
    # print(train_dataset)

    population = init_population(terminals) 

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
    test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]
    # Scale train fitnesses
    # train_fitnesses_scaled = scale(train_fitnesses)
    
    # Best of run
    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    # Save best train performance
    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    # Save test performance
    best_test_fit_list = [test_fitnesses[train_fitnesses.index(min(train_fitnesses))]]

    # Calculate complexity of each individual
    # pop_complexities = []
    # features_contributions_list = []
    # for ind in population:
    #     slope_value, features_dict = slope_based_complexity(ind, train_dataset)
    #     pop_complexities.append(slope_value)
    #     features_contributions_list.append(features_dict)
    #     # Scale complexities
    #     pop_complexities_scaled = scale(pop_complexities)

    # Calculate the variance of each individual
    variances = [compute_variance(ind, train_dataset) for ind in population]
    # Save the best of run variance
    best_of_run_variance = [variances[variances.index(min(variances))]]

    # Save complexities
    # iodc = [IODC(max_IODC, z, best_of_run, train_dataset)]
    # Sve best of run complexity
    # slope = [pop_complexities[train_fitnesses.index(min(train_fitnesses))]]
    # features_contribution = [features_contributions_list[train_fitnesses.index(min(train_fitnesses))]]

    # print('Best of run')
    # best_of_run.print_tree()
    # print('Best of run fitness:', best_of_run_f)
    # print('Best of run complexity:', slope[-1])
    # print('Best of run features contribution:', features_contribution[-1])

    # print('SLOPE:', slope)
    # print('FEATURES CONTRIBUTION:', features_contribution)

    # Save best ind size
    size = [best_of_run.size()]
    # Number of features used
    num_feats = [best_of_run.number_feats()]

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        new_pop=[deepcopy(best_of_run)]
    
        # print('LEN VARIANCES', len(variances))
        # print('LEN TRAIN FITNESSES', len(train_fitnesses))

        while len(new_pop) < POP_SIZE:

            
            prob = random()

            parent = tournament(population, train_fitnesses, variances)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses, variances)

                parent_orig = deepcopy(parent)
                parent2_orig = deepcopy(parent2)

                parent.crossover(parent2)

                # If children exceed parents
                if parent.depth() > MAX_DEPTH:
                    parent = choice([parent_orig, parent2_orig])
                
                if parent2.depth() > MAX_DEPTH:
                    parent2 = choice([parent_orig, parent2_orig])

                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')
                
                new_pop.append(parent)
                new_pop.append(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent_orig = deepcopy(parent)

                parent.mutation()

                if parent.depth() > MAX_DEPTH:
                    parent = parent2_orig

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')
                
                new_pop.append(parent)
            
            # NOTE: Replication may also occur if no condition is met
            else:
                new_pop.append(parent)

        new_train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in new_pop]

        # Check if len population exceeded
        if len(new_pop) > POP_SIZE:
            print('POP SIZE EXCEEDED')
            # Remove worst individual
            idx_worst = new_train_fitnesses.index(max(new_train_fitnesses))
            new_pop.pop(idx_worst)
            new_train_fitnesses.pop(idx_worst)
        
        if len(new_pop) != POP_SIZE:
            raise Exception('POP SIZE EXCEEDED!!!')
            
        population = deepcopy(new_pop)
        train_fitnesses = deepcopy(new_train_fitnesses)
        test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]

        if min(train_fitnesses) > best_of_run_f:
            raise Exception('Best individual fitness increased')
        
        best_of_run_f = min(train_fitnesses)
        best_of_run_gen = gen
        best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])        
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        best_test_fit_list.append(fitness(best_of_run, test_dataset, test_target))

        # pop_complexities = []
        # features_contributions_list = []
        # # Save complexities
        # for ind in population:
        #     slope_value, features_dict = slope_based_complexity(ind, train_dataset)
        #     pop_complexities.append(slope_value)
        #     pop_complexities_scaled = scale(pop_complexities)
        #     features_contributions_list.append(features_dict)

        # # Save best of run complexities
        # slope.append(pop_complexities[train_fitnesses.index(min(train_fitnesses))])
        # features_contribution.append(features_contributions_list[train_fitnesses.index(min(train_fitnesses))])

        # Calculate the variance of each individual
        variances = [compute_variance(ind, train_dataset) for ind in population]      
        # Save best of run variance  
        best_of_run_variance.append(variances[variances.index(min(variances))])

        # print('Best of run')
        # best_of_run.print_tree()
        # print('Used feats:', best_of_run.used_feats())
        # print('Best of run fitness:', best_of_run_f)
        # print('Best of run complexity:', slope[-1])
        # print('Best of run features contribution:', features_contribution[-1])

        # Save size
        size.append(best_of_run.size())

        # Number of unique feats
        num_feats.append(best_of_run.number_feats())
        # num_feats_distribution.append([ind.number_feats() for ind in population])
        # mean_number_feats.append(np.mean(num_feats_distribution[-1]))

        print('NEW BEST FINTESS', best_of_run_f)
        print('FITNESS IN TEST', best_test_fit_list[-1])
        print('BEST INDIVIDUAL VARIANCE:', best_of_run_variance[-1])
        
        # Optimal solution found
        if best_of_run_f == 0:
            break   

    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))

    return best_train_fit_list, best_test_fit_list, best_ind_list,\
        best_of_run_variance, \
        size, \
        num_feats
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
