from random import randint, random
from copy import deepcopy
import numpy as np
import time

from configs_OpEq import *
from gptree import GPTree

hist = {}

def init_target_hist(pop_hist_fitnesses, max_fitness):
    hist = {}

    nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))

    if TARGET == 'FLAT':

        bin_capacity = int(POP_SIZE / nr_bins)


        for i in range(1, nr_bins + 1):
            hist[i] = bin_capacity
    
    elif TARGET == 'DYN':

        # Fitnesses are normalized for a minimization problem
        all_fitnesses = {bin: np.mean(max_fitness - np.array(pop_hist_fitnesses[bin])) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}
        
        for bin in range(1, nr_bins + 1):
            hist[bin] = np.round(POP_SIZE *  (all_fitnesses[bin] / sum(all_fitnesses.values())))

        # TODO: CONFIRM THIS!
        while sum(hist.values()) > POP_SIZE:
            print('TARGET > POP SIZE')
            best_bin = max(hist, key = hist.get)
            hist[best_bin] -= 1
        
        while sum(hist.values()) < POP_SIZE:
            print('TARGET < POP SIZE')
            best_bin = max(hist, key = hist.get)
            hist[best_bin] += 1
    
    return hist

def update_target_hist(pop_hist_fitnesses, max_fitness):

    hist = {}

    # Fitnesses are normalized for a minimization problem   
    all_fitnesses = {bin: np.mean(max_fitness - np.array(pop_hist_fitnesses[bin])) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}
    
    for bin in pop_hist_fitnesses.keys():
        hist[bin] = int(np.round(POP_SIZE *  (all_fitnesses[bin] / sum(all_fitnesses.values()))))
    
    # TODO: CONFIRM THIS!
    while sum(hist.values()) > POP_SIZE:
        best_bin = max(hist, key = hist.get)
        hist[best_bin] -= 1
    
    while sum(hist.values()) < POP_SIZE:
        best_bin = max(hist, key = hist.get)
        hist[best_bin] += 1



    if sum(hist.values()) > POP_SIZE or sum(hist.values()) < POP_SIZE:
        raise Exception('TARGETS != POP SIZE. SUM =', sum(hist.values()))

    return hist

def reset_pop_hist(bins):

    hist = {}
    hist_fitnesses = {}

    for bin in bins:
        hist[bin] = 0
        hist_fitnesses[bin] = []

    
    return hist, hist_fitnesses

def init_hist(population, train_fitnesses):
    nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))

    pop_hist = {}
    pop_hist_fitness = {}

    for idx, ind in enumerate(population):
        ind_bin = ind.get_bin()

        if ind_bin in pop_hist.keys():
            pop_hist[ind_bin] += 1
            pop_hist_fitness[ind_bin].append(train_fitnesses[idx])
        else:
            pop_hist[ind_bin] = 1
            pop_hist_fitness[ind_bin] = [train_fitnesses[idx]]


    for i in range(1, nr_bins + 1):
        if i not in pop_hist.keys():
            pop_hist[i] = 0
            pop_hist_fitness[i] = []

    return pop_hist, pop_hist_fitness

def check_bin_capacity(target_hist, pop_hist, ind_bin, ind_fitness, best_of_run_f):
    """
    Check if individual can be added to the population given the ideal target distribution
    """

    # If in range
    if ind_bin in pop_hist.keys():
        # Bin not full
        if pop_hist[ind_bin] < target_hist[ind_bin]:
            # print('NOT FULL', pop_hist[ind_bin], '<', target_hist[ind_bin])
            return True
        # Full bin but best of run -> exceed capacity
        elif pop_hist[ind_bin] >= target_hist[ind_bin] and ind_fitness < best_of_run_f:
            # print('FULL BUT BEST OF RUN', ind_fitness, '<', best_of_run_f)
            return True
    # Out of range but best of run -> add new bin
    elif ind_fitness < best_of_run_f:
        # print('OUT OF RANGE BUT BEST OF RUN')
        # print(ind_fitness, '<', best_of_run_f)
        return True
    
    return False

def update_hist(target_hist, pop_hist, pop_hist_fitness, ind_bin, ind_fitness):
    """
    When individual is added to the population, update the population histogram
    and maybe the target population when the new bine xceed the old upper bound
    """

    # Check is exceeded. change target hist to have 1 between max and new max
    if ind_bin in pop_hist.keys():
        # print('ADD NEW IND to bin', ind_bin)
        # print('ADD INDIVIDUAL TO BIN')
        pop_hist[ind_bin] += 1
        pop_hist_fitness[ind_bin].append(ind_fitness)

        # print('NEW POP HIST', pop_hist)
        # print('NEW POP FITNESSES HIST', pop_hist_fitness)
        # print('NEW TARGET HIST', target_hist)
    
    else:
        # print('ADD NEW BINS UNTIL', ind_bin)
        # Add new bins
        for new_bin in range(max(target_hist.keys()) + 1, ind_bin + 1):
            target_hist[new_bin] = 1
            pop_hist[new_bin] = 0
            pop_hist_fitness[new_bin] = []

        pop_hist[ind_bin] = 1
        pop_hist_fitness[ind_bin].append(ind_fitness)

        # print('NEW POP HIST', pop_hist)
        # print('NEW POP FITNESSES HIST', pop_hist_fitness)
        # print('NEW TARGET HIST', target_hist)
    
    return target_hist, pop_hist, pop_hist_fitness

                   
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

    population = init_population(terminals) 

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    pop_hist, pop_hist_fitnesses = init_hist(population, train_fitnesses)

    target_hist = init_target_hist(pop_hist_fitnesses, max(train_fitnesses))

    for ind in population:
        print('SIZE', ind.size())

    print('INITIAL POP HIST', pop_hist)
    print('INITIAL POP FITNESS HIST', pop_hist_fitnesses)
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

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        # Reset population histogram
        pop_hist, pop_hist_fitnesses = reset_pop_hist(list(pop_hist.keys()))

        new_pop = []
        new_train_fitnesses = []

        while len(new_pop) < POP_SIZE:
            
            prob = random()

            parent = tournament(population, train_fitnesses)

            # Crossover
            if prob < XO_RATE:

                # print('CROSSOVER')
                parent2 = tournament(population, train_fitnesses)

                parent.crossover(parent2)

                parent_fitness = fitness(parent, train_dataset, train_target)
                parent2_fitness = fitness(parent2, train_dataset, train_target)


                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')
                
                if check_bin_capacity(target_hist, pop_hist, ind_bin = parent.get_bin(),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f):
                    


                    new_pop.append(parent)
                    target_hist, pop_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                
                if len(new_pop) < POP_SIZE and check_bin_capacity(target_hist, pop_hist, ind_bin = parent2.get_bin(),
                                                                  ind_fitness = parent2_fitness,
                                                                  best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent2)
                    target_hist, pop_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist, pop_hist_fitnesses, parent2.get_bin(), parent2_fitness)
                    new_train_fitnesses.append(parent2_fitness)

                    if parent2_fitness < best_of_run_f:
                        best_of_run_f = parent2_fitness
                        best_of_run = deepcopy(parent2)


            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent.mutation()

                parent_fitness = fitness(parent, train_dataset, train_target)

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')

                if check_bin_capacity(target_hist, pop_hist, ind_bin = parent.get_bin(),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f):
                    

                    new_pop.append(parent)
                    target_hist, pop_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
            
            # NOTE: Replication may also occur if no condition is met
            else:

                parent_fitness = fitness(parent, train_dataset, train_target)

                if check_bin_capacity(target_hist, pop_hist, ind_bin = parent.get_bin(),
                                      ind_fitness = fitness(parent, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f):
                    
                    new_pop.append(parent)
                    target_hist, pop_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist, pop_hist_fitnesses, parent.get_bin(), parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
            
        population = new_pop
        train_fitnesses = new_train_fitnesses

        print('NEW TARGET HIST')
        print(target_hist)

        print('NEW POP HIST')
        print(pop_hist)

        # print('NEW POP FITNESS HIST')
        # print(pop_hist_fitnesses)

        target_hist = update_target_hist(pop_hist_fitnesses, max(train_fitnesses)) # TODO

        # train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]        
           
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
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen
    
