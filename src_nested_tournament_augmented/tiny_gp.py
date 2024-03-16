from random import randint, random, choice
from copy import deepcopy
import numpy as np
import time

from configs_nested import *
from gptree import GPTree
from complexity_measures import init_IODC, IODC, mean_IODC
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
    
                
def tournament(population, fitnesses):
    """
    Tournament selection
    """
    # Select random individuals to compete
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_FITNESS)]

    # Get their fitness values
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_FITNESS)]
    
    # Return the winner
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 

def nested_tournament(population, fitnesses, dataset):

    rand_nmb = random()

    if rand_nmb < DTOURNM_PROBABILITY:

        winners = []

        # Choose N first based on RMSE
        for _ in range(TOURNAMENT_COMPLEXITY):
            winners.append(tournament(population, fitnesses))

        slopes = []

        # Calculate complexities
        for ind in winners:
            slope_value, _ = slope_based_complexity(ind, dataset)
            slopes.append(slope_value)
            
        # Choose the one with lowest slope based complexity
        return deepcopy(winners[slopes.index(min(slopes))])

    else:
        return tournament(population, fitnesses)
            
def evolve(train_dataset, test_dataset, sampled_dataset, train_target, test_target, terminals):

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

    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    # Save best train performance
    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    # Save test performance
    best_test_fit_list = [test_fitnesses[train_fitnesses.index(min(train_fitnesses))]]
    # Save mean train performance
    mean_train_fit_list = [np.mean(train_fitnesses)]
    mean_test_fit_list = [np.mean(test_fitnesses)]
    # Save complexities
    # iodc = [IODC(max_IODC, z, best_of_run, train_dataset)]
    slope_value, features_dict = slope_based_complexity(best_of_run, train_dataset)
    slope = [slope_value]
    features_contribution = [features_dict]

    # print('SLOPE:', slope)
    # print('FEATURES CONTRIBUTION:', features_contribution)

    slope_sampled_value, features_dict_sampled = slope_based_complexity(best_of_run, sampled_dataset)
    slope_sampled = [slope_sampled_value]
    features_contribution_sampled = [features_dict_sampled]

    slope_test_value, features_dict_test = slope_based_complexity(best_of_run, test_dataset)
    slope_test = [slope_test_value]
    features_contribution_test = [features_dict_test]

    # print('SLOPE sampled:', slope)
    # print('FEATURES CONTRIBUTION sampled:', features_contribution)

    # Save complexity distributions
    # iodc_distribution = [[IODC(max_IODC, z, ind, train_dataset) for ind in population]]
    # slope_distribution = [[slope_based_complexity(ind, train_dataset) for ind in population]]
    # slope_sampled_distribution = [[slope_based_complexity(ind, sampled_dataset) for ind in population]]
    # Save mean complexities
    # mean_iodc = [np.mean(iodc_distribution[0])]
    # mean_slope = [np.mean(slope_distribution[0])]
    # mean_slope_sampled = [np.mean(slope_sampled_distribution[0])]
    # Save best ind size
    size = [best_of_run.size()]
    # Save sizes
    size_distribution = [[ind.size() for ind in population]]
    # Save mean sizes
    mean_size = [np.mean(size_distribution[0])]
    # Save measure of interpretability
    # Number of ops
    no = [best_of_run.number_operations()]
    no_distribution = [[ind.number_operations() for ind in population]]
    mean_no = [np.mean(no_distribution[0])]
    # Number of features used
    num_feats = [best_of_run.number_feats()]
    num_feats_distribution = [[ind.number_feats() for ind in population]]
    mean_number_feats = [np.mean(num_feats_distribution[-1])]

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        new_pop=[deepcopy(best_of_run)]

        while len(new_pop) < POP_SIZE:
            
            prob = random()

            parent = nested_tournament(population, train_fitnesses, sampled_dataset)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = nested_tournament(population, train_fitnesses, sampled_dataset)

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
            
        population = new_pop.copy()
        train_fitnesses = new_train_fitnesses.copy()
        test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]

        # UNCOMMENT HERE TO GET BEST SO FAR
        # if min(train_fitnesses) < best_of_run_f:

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
        mean_train_fit_list.append(np.mean(train_fitnesses))
        mean_test_fit_list.append(np.mean(test_fitnesses))

        # Save complexities
        # iodc.append(IODC(max_IODC, z, best_of_run, train_dataset))
        slope_value, features_dict = slope_based_complexity(best_of_run, train_dataset)
        slope.append(slope_value)
        features_contribution.append(features_dict)

        # print('SLOPE AFTER GEN:', slope)
        # print('FEATURES CONTRIBUTION AFTER GEN:', features_contribution)

        slope_sampled_value, features_dict_sampled = slope_based_complexity(best_of_run, sampled_dataset)
        slope_sampled.append(slope_sampled_value)
        features_contribution_sampled.append(features_dict_sampled)

        slope_test_value, features_dict_test = slope_based_complexity(best_of_run, test_dataset)
        slope_test.append(slope_test_value)
        features_contribution_test.append(features_dict_test)

        # print('SLOPE sampled AFTER GEN:', slope_sampled)
        # print('FEATURES sampled CONTRIBUTION AFTER GEN:', features_contribution_sampled)
        # Save complexity distributions
        # iodc_distribution.append([IODC(max_IODC, z, ind, train_dataset) for ind in population])
        # slope_distribution.append([slope_based_complexity(ind, train_dataset) for ind in population])
        # slope_sampled_distribution.append([slope_based_complexity(ind, sampled_dataset) for ind in population])
        # # Save mean complexities
        # # mean_iodc.append(np.mean(iodc_distribution[-1]))
        # mean_slope.append(np.mean(slope_distribution[-1]))
        # mean_slope_sampled.append(np.mean(slope_sampled_distribution[-1]))
        # Save size
        size.append(best_of_run.size())
        # Save size distribution
        size_distribution.append([ind.size() for ind in population])
        # Save mean size
        mean_size.append(np.mean(size_distribution[-1]))

        # Save iterpretability
        # Number of ops
        no.append(best_of_run.number_operations())
        no_distribution.append([ind.number_operations() for ind in population])
        mean_no.append(np.mean(no_distribution[-1]))
        # Number of unique feats
        num_feats.append(best_of_run.number_feats())
        num_feats_distribution.append([ind.number_feats() for ind in population])
        mean_number_feats.append(np.mean(num_feats_distribution[-1]))

        print('NEW BEST FINTESS', best_of_run_f)
        print('FITNESS IN TEST', best_test_fit_list[-1])
        print('BEST INDIVIDUAL COMPLEXITY:', slope[-1])
        print('BEST INDIVIDUAL sampled COMPLEXITY:', slope_sampled[-1])
        
        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, \
        mean_train_fit_list, mean_test_fit_list, \
        slope, \
        slope_sampled, \
        slope_test, \
        features_contribution, features_contribution_sampled, features_contribution_test, \
        size, mean_size, size_distribution, \
        no, mean_no, no_distribution, \
        num_feats, mean_number_feats, num_feats_distribution
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
