from random import randint, random, choice
from copy import deepcopy
import numpy as np
import time

from configs_variance import *
from gptree import GPTree
from complexity_measures import init_slope_based_complexity, slope_based_complexity, mean_slope_based_complexity
                   
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


def tournament(population, fitnesses, complexities):
    """
    Tournament selection: modified tournament selection scheme
    """
    # Select random individuals to compete
    tournament_idxs = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]
    
    # Get their complexity values
    complexity_values = [complexities[ind_idx] for ind_idx in tournament_idxs]
    # Get their fitness values
    tournament_fitnesses = [fitnesses[ind_idx] for ind_idx in tournament_idxs]

    # A dominates over B
    if (tournament_fitnesses[0] <= tournament_fitnesses[1] and complexity_values[0] < complexity_values[1]) or \
        (tournament_fitnesses[0] < tournament_fitnesses[1] and complexity_values[0] <= complexity_values[1]):
        return deepcopy(population[tournament_idxs[0]])
    # B dominates over A
    elif (tournament_fitnesses[1] <= tournament_fitnesses[0] and complexity_values[1] < complexity_values[0]) or \
        (tournament_fitnesses[1] < tournament_fitnesses[0] and complexity_values[1] <= complexity_values[0]):
        return deepcopy(population[tournament_idxs[1]])
    # No dominance. Rectilinear distance of A is smaller than euclidean distance of B
    elif abs(tournament_fitnesses[0] + complexity_values[0]) < np.sqrt(tournament_fitnesses[1]**2 + complexity_values[1]**2):
        return deepcopy(population[tournament_idxs[0]])
    # No dominance. Rectilinear distance of B is smaller than euclidean distance of A
    elif abs(tournament_fitnesses[1] + complexity_values[1]) < np.sqrt(tournament_fitnesses[0]**2 + complexity_values[0]**2):
        return deepcopy(population[tournament_idxs[1]])
    # Return individual with smaller complexity
    else:
        return deepcopy(population[tournament_idxs[complexity_values.index(min(complexity_values))]]) 


def scale(values):
    """
    Scale values to [0, 1]
    """
    min_val = min(values)
    max_val = max(values)
    
    # If all values are the same, return a list of zeros
    if min_val == max_val:
        return [0 for _ in values]
    
    return [(val - min_val) / (max_val - min_val) for val in values]


def evolve(train_dataset, test_dataset, train_target, test_target, terminals):

    # print('TRAIN DATASET')
    # print(train_dataset)

    population = init_population(terminals) 

    # for ind in population:
    #     ind.print_tree()
    #     print(ind.depth())
        # print(ind.number_operations())
        # print(len(set(ind.used_features())))
        # print('----------------------')
 

    # Upper bounds for complexities
    # z, max_IODC = init_IODC(train_dataset, train_target)
    # max_slope_complexity = init_slope_based_complexity(train_dataset, train_target)

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
    test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]
    # Scale train fitnesses
    train_fitnesses_scaled = scale(train_fitnesses)
    
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
    pop_complexities = []
    features_contributions_list = []
    for ind in population:
        slope_value, features_dict = slope_based_complexity(ind, train_dataset)
        pop_complexities.append(slope_value)
        features_contributions_list.append(features_dict)
        # Scale complexities
        pop_complexities_scaled = scale(pop_complexities)

    # Save mean train performance
    # mean_train_fit_list = [np.mean(train_fitnesses)]
    # mean_test_fit_list = [np.mean(test_fitnesses)]
    # Save complexities
    # iodc = [IODC(max_IODC, z, best_of_run, train_dataset)]
    # Sve best of run complexity
    slope = [pop_complexities[train_fitnesses.index(min(train_fitnesses))]]
    features_contribution = [features_contributions_list[train_fitnesses.index(min(train_fitnesses))]]

    # print('Best of run')
    # best_of_run.print_tree()
    # print('Best of run fitness:', best_of_run_f)
    # print('Best of run complexity:', slope[-1])
    # print('Best of run features contribution:', features_contribution[-1])

    # print('SLOPE:', slope)
    # print('FEATURES CONTRIBUTION:', features_contribution)

    # slope_test_value, features_dict_test = slope_based_complexity(best_of_run, test_dataset)
    # slope_test = [slope_test_value]
    # features_contribution_test = [features_dict_test]

    # print('SLOPE AUGMENTED:', slope)
    # print('FEATURES CONTRIBUTION AUGMENTED:', features_contribution)

    # Save complexity distributions
    # iodc_distribution = [[IODC(max_IODC, z, ind, train_dataset) for ind in population]]
    # slope_distribution = [[slope_based_complexity(ind, train_dataset) for ind in population]]
    # slope_augmented_distribution = [[slope_based_complexity(ind, augmented_dataset) for ind in population]]
    # Save mean complexities
    # mean_iodc = [np.mean(iodc_distribution[0])]
    # mean_slope = [np.mean(slope_distribution[0])]
    # mean_slope_augmented = [np.mean(slope_augmented_distribution[0])]
    # Save best ind size
    size = [best_of_run.size()]
    # Save sizes
    # size_distribution = [[ind.size() for ind in population]]
    # # Save mean sizes
    # mean_size = [np.mean(size_distribution[0])]
    # Save measure of interpretability
    # Number of ops
    # no = [best_of_run.number_operations()]
    # no_distribution = [[ind.number_operations() for ind in population]]
    # mean_no = [np.mean(no_distribution[0])]
    # Number of features used
    num_feats = [best_of_run.number_feats()]
    # num_feats_distribution = [[ind.number_feats() for ind in population]]
    # mean_number_feats = [np.mean(num_feats_distribution[-1])]

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        new_pop=[deepcopy(best_of_run)]
    
        # print('LEN POP COMPLEXITIES SCALED', len(pop_complexities_scaled))
        # print('LEN TRAIN FITNESSES SCALED', len(pop_complexities_scaled))

        while len(new_pop) < POP_SIZE:

            
            prob = random()

            parent = tournament(population, train_fitnesses_scaled, pop_complexities_scaled)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses_scaled, pop_complexities_scaled)

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
        train_fitnesses_scaled = scale(train_fitnesses)
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

        # Save mean train performance
        # mean_train_fit_list.append(np.mean(train_fitnesses))
        # mean_test_fit_list.append(np.mean(test_fitnesses))

        pop_complexities = []
        features_contributions_list = []
        # Save complexities
        for ind in population:
            slope_value, features_dict = slope_based_complexity(ind, train_dataset)
            pop_complexities.append(slope_value)
            pop_complexities_scaled = scale(pop_complexities)
            features_contributions_list.append(features_dict)

        # Save best of run complexities
        slope.append(pop_complexities[train_fitnesses.index(min(train_fitnesses))])
        features_contribution.append(features_contributions_list[train_fitnesses.index(min(train_fitnesses))])

        # print('Best of run')
        # best_of_run.print_tree()
        # print('Used feats:', best_of_run.used_feats())
        # print('Best of run fitness:', best_of_run_f)
        # print('Best of run complexity:', slope[-1])
        # print('Best of run features contribution:', features_contribution[-1])

        # print('SLOPE and FEATS CONTROBUTION:', slope_based_complexity(best_of_run, train_dataset))

        # print('SLOPE AFTER GEN:', slope)
        # print('FEATURES CONTRIBUTION AFTER GEN:', features_contribution)

        # slope_test_value, features_dict_test = slope_based_complexity(best_of_run, test_dataset)
        # slope_test.append(slope_test_value)
        # features_contribution_test.append(features_dict_test)

        # print('SLOPE AUGMENTED AFTER GEN:', slope_augmented)
        # print('FEATURES AUGMENTED CONTRIBUTION AFTER GEN:', features_contribution_augmented)
        # Save complexity distributions
        # iodc_distribution.append([IODC(max_IODC, z, ind, train_dataset) for ind in population])
        # slope_distribution.append([slope_based_complexity(ind, train_dataset) for ind in population])
        # slope_augmented_distribution.append([slope_based_complexity(ind, augmented_dataset) for ind in population])
        # # Save mean complexities
        # # mean_iodc.append(np.mean(iodc_distribution[-1]))
        # mean_slope.append(np.mean(slope_distribution[-1]))
        # mean_slope_augmented.append(np.mean(slope_augmented_distribution[-1]))
        # Save size
        size.append(best_of_run.size())
        # # Save size distribution
        # size_distribution.append([ind.size() for ind in population])
        # # Save mean size
        # mean_size.append(np.mean(size_distribution[-1]))

        # Save iterpretability
        # # Number of ops
        # no.append(best_of_run.number_operations())
        # no_distribution.append([ind.number_operations() for ind in population])
        # mean_no.append(np.mean(no_distribution[-1]))
        # Number of unique feats
        num_feats.append(best_of_run.number_feats())
        # num_feats_distribution.append([ind.number_feats() for ind in population])
        # mean_number_feats.append(np.mean(num_feats_distribution[-1]))

        print('NEW BEST FINTESS', best_of_run_f)
        print('FITNESS IN TEST', best_test_fit_list[-1])
        print('BEST INDIVIDUAL COMPLEXITY:', slope[-1])
        
        # Optimal solution found
        if best_of_run_f == 0:
            break   

    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))

    return best_train_fit_list, best_test_fit_list, best_ind_list,\
        slope, \
        features_contribution, \
        size, \
        num_feats
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
