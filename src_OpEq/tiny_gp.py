from random import randint, random, choice
from copy import deepcopy
import numpy as np
import time

from configs_OpEq import *
from gptree import GPTree
from opEq import init_target_hist, init_hist, update_target_hist, reset_pop_hist, check_bin_capacity, update_hist, get_population_len_histogram, get_best_ind_in_bins
                   
def init_population(terminals):
    """
    Ramped half-and-half initialization
    """

    # Number of individuals of each depth and initialized with each method
    inds_per_depth = int((POP_SIZE / (MAX_INITIAL_DEPTH - 1)) / 2)

    print(inds_per_depth)

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

def lexitournament(population, fitnesses):
    """
    Lexigographic Parsimony Tournament
    """

    # Select random individuals to compete
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)] # indexes of individuals

    # Get their fitness values
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)] # fitnesses of individuals

    min_indices = [index for index, value in enumerate(tournament_fitnesses) if value == min(tournament_fitnesses)]

    if len(min_indices) == 1:
        return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 
    else:
        print('TIE IN TOURNAMENT!!')
        min_size = 9999999
        smallest_ind = None
        for idx in min_indices:
            ind = population[tournament[idx]]
            print('IND SIZE:', ind.size())
            if ind.size() < min_size:
                min_size = ind.size()
                smallest_ind = ind

        print('WINNER SIZE:', smallest_ind.size())
        return deepcopy(smallest_ind)


            
def evolve(train_dataset, test_dataset, train_target, test_target, terminals):

    population = init_population(terminals) 

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    pop_hist_fitnesses = init_hist(population, train_fitnesses)

    target_hist = init_target_hist(pop_hist_fitnesses, max(train_fitnesses))

    # for ind in population:
    #     print('IND:')
    #     ind.print_tree()
    #     print('DEPTH', ind.depth())
    #     print('SIZE', ind.size())

    print('INITIAL POP FITNESS HIST')
    for key, value in pop_hist_fitnesses.items():
        print(f"{key}: {len(value)}", end=' ')
    print()

    print('TARGET HIST', target_hist)

    # test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]

    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    # Save best train performance
    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    # Save test performance
    best_test_fit_list = [fitness(best_of_run, test_dataset, test_target)]
    # Save best program size
    best_size = [best_of_run.size()]
    # Save mean size
    mean_size = [np.mean([ind.size() for ind in population])]
    # Save distributions
    target_histogram = [[target_hist[key] for key in sorted(target_hist.keys())]]
    #population_histogram = [[pop_hist_fitnesses[key] for key in sorted(pop_hist_fitnesses.keys())]]
    population_histogram = [get_population_len_histogram(pop_hist_fitnesses)]

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        # Reset population histogram
        pop_hist_fitnesses = reset_pop_hist(list(pop_hist_fitnesses.keys()))

        new_pop = []
        new_train_fitnesses = []

        while len(new_pop) < POP_SIZE:
            
            prob = random()

            parent = tournament(population, train_fitnesses)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses)

                parent_orig = deepcopy(parent)
                parent2_orig = deepcopy(parent)

                parent.crossover(parent2)

                # if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                #     print('Crossover generated an individual that exceeds depth.')
                #     print('Child 1 depth:', parent.depth())
                #     print('Child 2 depth:', parent2.depth())

                # If children exceed parents
                if parent.depth() > MAX_DEPTH:
                    parent = choice([parent_orig, parent2_orig])
                
                if parent2.depth() > MAX_DEPTH:
                    parent2 = choice([parent_orig, parent2_orig])

                parent_fitness = fitness(parent, train_dataset, train_target)
                parent2_fitness = fitness(parent2, train_dataset, train_target)

                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')
                
                # First child
                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = parent.get_bin(),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f):
                    


                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen
                
                # Second child
                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = parent2.get_bin(),
                                                                  ind_fitness = parent2_fitness,
                                                                  best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent2)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, parent2.get_bin(), parent2_fitness)
                    new_train_fitnesses.append(parent2_fitness)

                    if parent2_fitness < best_of_run_f:
                        best_of_run_f = parent2_fitness
                        best_of_run = deepcopy(parent2)
                        best_of_run_gen = gen

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent_orig = deepcopy(parent)

                parent.mutation()

                parent_fitness = fitness(parent, train_dataset, train_target)

                # If children exceed parents
                if parent.depth() > MAX_DEPTH:
                    parent = parent_orig

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')

                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = parent.get_bin(),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen

            # NOTE: Replication may also occur if no condition is met
            else:

                parent_fitness = fitness(parent, train_dataset, train_target)

                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = parent.get_bin(),
                                      ind_fitness = fitness(parent, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f):
                    
                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen
        
        # Check if len population exceeded
        if len(new_pop) > POP_SIZE:
            print('POP SIZE EXCEEDED')
            # Remove worst individual
            idx_worst = new_train_fitnesses.index(max(new_train_fitnesses))
            new_pop.pop(idx_worst)
            new_train_fitnesses.pop(idx_worst)
        
        if len(new_pop) != POP_SIZE:
            raise Exception('POP SIZE EXCEEDED!!!')

        population = new_pop
        train_fitnesses = new_train_fitnesses

        # print('NEW POP FITNESS HIST')
        # print(pop_hist_fitnesses)

        target_hist = update_target_hist(pop_hist_fitnesses, max(train_fitnesses)) # TODO

        print('POP HIST')
        for key, value in pop_hist_fitnesses.items():
            print(f"{key}: {len(value)}", end=' ')
        print()

        print('NEW TARGET HIST')
        print(target_hist)

        # train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]        
           
        # print("________________________")
        # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
        # best_of_run.print_tree()
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        best_test_fit_list.append(fitness(best_of_run, test_dataset, test_target))
        # Save beat size
        best_size.append(best_of_run.size())
        # Save mean size
        mean_size.append(np.mean([ind.size() for ind in population]))
        # Save distributions
        target_histogram.append([target_hist[key] for key in sorted(target_hist.keys())])
        #population_histogram.append([pop_hist_fitnesses[key] for key in sorted(pop_hist_fitnesses.keys())])
        population_histogram.append(get_population_len_histogram(pop_hist_fitnesses))

        # print(get_best_ind_in_bins(pop_hist_fitnesses))

        print('BIN OF BEST INDIVIDUAL', best_of_run.get_bin())

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, best_size, mean_size, target_histogram, population_histogram
    
