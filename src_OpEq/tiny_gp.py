from random import randint, random
from copy import deepcopy
import numpy as np
import time

from configs_OpEq import *
from gptree import GPTree

hist = {}

def init_target_hist():

    nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))
    bin_capacity = POP_SIZE / nr_bins

    hist = {}

    for i in range(1, nr_bins + 1):
        hist[i] = bin_capacity
    
    return hist

def reset_pop_hist(nr_bins):

    hist = {}

    for i in range(1, nr_bins + 1):
        hist[i] = 0
    
    return hist

def init_hist(population):
    nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))

    pop_hist = {}

    for ind in population:
        ind_bin = ind.get_bin()

        if ind_bin in pop_hist.keys():
            pop_hist[ind_bin] += 1
        else:
            pop_hist[ind_bin] = 1

    for i in range(1, nr_bins + 1):
        if i not in pop_hist.keys():
            pop_hist[i] = 0

    return pop_hist

def check_bin_capacity(target_hist, pop_hist, ind_bin, ind_fitness, best_of_run_f):
    """
    Check if individual can be added to the population given the ideal target distribution
    """
    # If in range
    if ind_bin in pop_hist.keys():
        # Bin not full
        if pop_hist[ind_bin] < target_hist[ind_bin]:
            return True
        # Full bin but best of run -> exceed capacity
        elif pop_hist[ind_bin] >= target_hist[ind_bin] and ind_fitness < best_of_run_f:
            return True
    # Out of range but best of run -> add new bin
    elif ind_fitness < best_of_run_f:
        return True
    
    return False

def update_hist(target_hist, pop_hist, ind_bin):
    """
    When individual is added to the population, update the population histogram
    and maybe the target population when the new bine xceed the old upper bound
    """
    # Check is exceeded. change target hist to have 1 between max and new max
    if ind_bin in pop_hist.keys():
        pop_hist[ind_bin] += 1
    
    else:
        # Add new bins
        for new_bin in range(max(target_hist.keys()) + 1, ind_bin + 1):
            target_hist[new_bin] = 1
            pop_hist[new_bin] = 0

        pop_hist[ind_bin] = 1
    
    return pop_hist, target_hist

                   
def init_population(terminals):
    """
    Ramped half-and-half initialization
    """

    # Number of individuals of each depth and initialized with each method
    inds_per_depth = int((POP_SIZE / (MAX_INITIAL_DEPTH + 1)) / 2)

    pop = []
    for max_depth in range(MIN_DEPTH, MAX_INITIAL_DEPTH + 1):
        
        # Grow
        for _ in range(inds_per_depth):
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = True, max_depth = max_depth)
            ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

            pop.append(ind) 
        
        # Full
        for _ in range(inds_per_depth):
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = False, max_depth = max_depth)  
            ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

            pop.append(ind) 


    # Edge case
    while len(pop) != POP_SIZE:
        # Generate random tree with random method to fill population
        max_depth = randint(MIN_DEPTH, MAX_INITIAL_DEPTH)
        grow = True if random() < 0.5 else False
        ind = GPTree(terminals = terminals)
        ind.random_tree(grow = grow, max_depth = max_depth)
        ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!
        pop.append(ind) 

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
            
def evolve(train_dataset, test_dataset, train_target, test_target, terminals):

    target_hist = init_target_hist()

    population = init_population(terminals) 

    pop_hist = init_hist(population)

    for ind in population:
        print('SIZE', ind.size())

    print(pop_hist)

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
    # test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]

    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    # Save best train performance
    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    # Save test performance
    best_test_fit_list = [fitness(best_of_run, test_dataset, test_target)]

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        # Reset population histogram
        pop_hist = reset_pop_hist(len(pop_hist.keys()))
        

        new_pop=[]

        while len(new_pop) < POP_SIZE:
            
            prob = random()

            parent = tournament(population, train_fitnesses)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses)

                parent.crossover(parent2)

                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')
                
                if check_bin_capacity(target_hist, pop_hist, ind_size = parent.get_bin(),
                                      ind_fitness = fitness(ind, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f):


                    new_pop.append(parent)
                    target_hist, pop_hist = update_hist(target_hist, pop_hist, parent.get_bin())

                if len(new_pop) < POP_SIZE and check_bin_capacity(target_hist, pop_hist, ind_size = parent2.get_bin(),
                                                                  ind_fitness = fitness(ind, train_dataset, train_target),
                                                                  best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent2)
                    target_hist, pop_hist = update_hist(target_hist, pop_hist, parent2.get_bin())

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent.mutation()

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')
                
                if check_bin_capacity(target_hist, pop_hist, ind_size = parent.get_bin(),
                                      ind_fitness = fitness(ind, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent)
                    target_hist, pop_hist = update_hist(target_hist, pop_hist, parent.get_bin())
            
            # NOTE: Replication may also occur if no condition is met
            else:
                if check_bin_capacity(target_hist, pop_hist, ind_size = parent.get_bin(),
                                      ind_fitness = fitness(ind, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f):
                    
                    new_pop.append(parent)
                    target_hist, pop_hist = update_hist(target_hist, pop_hist, parent.get_bin())
            
        population = new_pop

        # hist = recalculate_target_hist() # TODO

        train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
        
        if min(train_fitnesses) < best_of_run_f:
            best_of_run_f = min(train_fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])        
           
        # print("________________________")
        # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
        # best_of_run.print_tree()
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        best_test_fit_list.append(fitness(best_of_run, test_dataset, test_target))

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen
    
