from random import randint, random, choice
from copy import deepcopy
import numpy as np
import time

from configs_bounded import *
from gptree import GPTree
from complexity_measures import init_IODC, IODC, mean_IODC
from complexity_measures import init_slope_based_complexity, slope_based_complexity, mean_slope_based_complexity
                   
def init_population(terminals):
    """
    Ramped half-and-half initialization
    """

    # Number of individuals of each depth and initialized with each method
    inds_per_depth = int((POP_SIZE / (MAX_INITIAL_DEPTH - 1)) / 2)

    pop = []
    pop_str = []

    for max_depth in range(MIN_DEPTH, MAX_INITIAL_DEPTH):

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
                ind.random_tree(grow = True, max_depth = max_depth)  
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
            ind.random_tree(grow = 1, max_depth = MAX_INITIAL_DEPTH)
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
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]

    # Get their fitness values
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]
    
    # Return the winner
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 
            
def evolve(train_dataset, val_dataset, test_dataset, train_val_dataset, train_target, test_target, terminals):

    # print('TRAIN DATASET')
    # print(train_dataset)

    # print('TRAIN SHAPE:', train_dataset.shape)
    # print('TEST SHAPE:', test_dataset.shape)
    # print('VAL SHAPE:', val_dataset.shape)
    # print('TRAIN VAL SHAPE:', train_val_dataset.shape)

    # print('Train head', train_dataset[:2])
    # print('Val tail', train_dataset[:-3])

    # print('Train val head', train_val_dataset[:2])
    # print('Train val tail', train_val_dataset[:-3])

    population = init_population(terminals) 

    # for ind in population:
    #     ind.print_tree()
    #     print(ind.number_operations())
    #     print(len(set(ind.used_features())))
    #     print('----------------------')
 

    # Upper bounds for complexities
    max_slope_complexity = init_slope_based_complexity(train_dataset, train_target)

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
    test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]

    print('Fitness done')

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
    slope = [slope_based_complexity(max_slope_complexity, best_of_run, train_dataset)]
    slope_val = [slope_based_complexity(max_slope_complexity, best_of_run, val_dataset)]
    slope_all = [slope_based_complexity(max_slope_complexity, best_of_run, train_val_dataset)]
    # Save complexity distributions
    slope_distribution = [[slope_based_complexity(max_slope_complexity, ind, train_dataset) for ind in population]]
    slope_val_distribution = [[slope_based_complexity(max_slope_complexity, ind, val_dataset) for ind in population]]
    # Save mean complexities
    mean_slope = [np.mean(slope_distribution[0])]
    mean_val_slope = [np.mean(slope_val_distribution[0])]

    print('Complexities done')

    # Save best ind size
    size = [best_of_run.size()]
    # Save sizes
    size_distribution = [[ind.size() for ind in population]]
    # Save mean sizes
    mean_size = [np.mean(size_distribution[0])]

    print('Size done')

    # # Save measure of interpretability
    # # Number of ops
    # no = [best_of_run.number_operations()]
    # no_distribution = [[ind.number_operations() for ind in population]]
    # mean_no = [np.mean(no_distribution[0])]

    # print('Number ops done')

    # # Number of features used
    # num_feats = [best_of_run.number_feats()]
    # num_feats_distribution = [[ind.number_feats() for ind in population]]
    # mean_number_feats = [np.mean(num_feats_distribution[-1])]

    # print('Number feats done')

    complexity_bound = None

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        # Save best of run maximum complexity on generation GENERATION BOUND
        if gen == GENERATION_BOUND:
            complexity_bound = slope_based_complexity(max_slope_complexity, best_of_run, train_dataset)

        new_pop=[]

        while len(new_pop) < POP_SIZE:
            
            prob = random()

            parent = tournament(population, train_fitnesses)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses)

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
                
                # Only add individual if it doesn't exceed maximum complexity
                if gen < GENERATION_BOUND:
                    new_pop.append(parent)
                elif gen >= GENERATION_BOUND and slope_based_complexity(max_slope_complexity, parent, train_dataset) < complexity_bound:
                    new_pop.append(parent)

                if gen < GENERATION_BOUND:
                    new_pop.append(parent2)
                elif gen >= GENERATION_BOUND and slope_based_complexity(max_slope_complexity, parent2, train_dataset) < complexity_bound:
                    new_pop.append(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent_orig = deepcopy(parent)

                parent.mutation()

                if parent.depth() > MAX_DEPTH:
                    parent = parent2_orig

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')
                
                if gen < GENERATION_BOUND:
                    new_pop.append(parent)
                elif gen >= GENERATION_BOUND and slope_based_complexity(max_slope_complexity, parent, train_dataset) < complexity_bound:
                    new_pop.append(parent)
            
            # NOTE: Replication may also occur if no condition is met
            else:

                if gen < GENERATION_BOUND:
                    new_pop.append(parent)
                elif gen >= GENERATION_BOUND and slope_based_complexity(max_slope_complexity, parent, train_dataset) < complexity_bound:
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
        
        if min(train_fitnesses) < best_of_run_f:
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
        slope.append(slope_based_complexity(max_slope_complexity, best_of_run, train_dataset))
        slope_val.append(slope_based_complexity(max_slope_complexity, best_of_run, val_dataset))
        slope_all.append(slope_based_complexity(max_slope_complexity, best_of_run, train_val_dataset))
        # Save complexity distributions
        slope_distribution.append([slope_based_complexity(max_slope_complexity, ind, train_dataset) for ind in population])
        slope_val_distribution.append([slope_based_complexity(max_slope_complexity, ind, val_dataset) for ind in population])

        # Save mean complexities
        mean_slope.append(np.mean(slope_distribution[-1]))
        mean_val_slope.append(np.mean(slope_val_distribution[-1]))

        # Save size
        size.append(best_of_run.size())
        # Save size distribution
        size_distribution.append([ind.size() for ind in population])
        # Save mean size
        mean_size.append(np.mean(size_distribution[-1]))

        # # Save iterpretability
        # # Number of ops
        # no.append(best_of_run.number_operations())
        # no_distribution.append([ind.number_operations() for ind in population])
        # mean_no.append(np.mean(no_distribution[-1]))
        # # Number of unique feats
        # num_feats.append(best_of_run.number_feats())
        # num_feats_distribution.append([ind.number_feats() for ind in population])
        # mean_number_feats.append(np.mean(num_feats_distribution[-1]))

        print('NEW BEST FINTESS', best_of_run_f)
        print('FITNESS IN TEST', best_test_fit_list[-1])
        
        # print('IODC DISTRIBUTION', iodc_distribution[-1])

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, \
        mean_train_fit_list, mean_test_fit_list, \
        slope, mean_slope, slope_distribution, \
        slope_val, mean_val_slope, slope_val_distribution, \
        slope_all, \
        size, mean_size, size_distribution
        # no, mean_no, no_distribution, \
        # num_feats, mean_number_feats, num_feats_distribution
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
