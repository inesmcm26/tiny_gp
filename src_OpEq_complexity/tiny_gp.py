from random import randint, random
from copy import deepcopy
import numpy as np

from configs_OpEq_complexity import *
from gptree import GPTree
from complexity_measures_new import init_IODC, IODC
from opEq import init_target_hist, init_hist, update_target_hist, reset_pop_hist, check_bin_capacity, update_hist, get_population_len_histogram, get_bin

def init_population(terminals, max_IODC, z, train_dataset):
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
            
            if IODC(max_IODC, z, ind, train_dataset) < MAX_COMPLEXITY:
                pop.append(ind) 
        
        # Full
        for _ in range(inds_per_depth):
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = False, max_depth = max_depth)  
            ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!
            
            if IODC(max_IODC, z, ind, train_dataset) < MAX_COMPLEXITY:
                pop.append(ind) 


    # Edge case
    while len(pop) != POP_SIZE:
        # Generate random tree with random method to fill population
        max_depth = randint(MIN_DEPTH, MAX_INITIAL_DEPTH)
        grow = True if random() < 0.5 else False
        ind = GPTree(terminals = terminals)
        ind.random_tree(grow = grow, max_depth = max_depth)
        ind.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!
        
        if IODC(max_IODC, z, ind, train_dataset) < MAX_COMPLEXITY:
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

    # Initialize IODC calculation
    z, max_IODC = init_IODC(train_dataset, train_target)
    
    population = init_population(terminals, max_IODC, z, train_dataset)

    # Initialize number of iterations with no val improvement
    nr_gen_no_improvement = 0

    # Calculate initial train fitnesses
    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    # Calculate initial population and target histograms
    pop_hist_fitnesses, bin_width = init_hist(population, train_fitnesses, max_IODC, z, train_dataset)

    target_hist = init_target_hist(pop_hist_fitnesses, max(train_fitnesses))

    for ind in population:
        ind_complexity = IODC(max_IODC, z, ind, train_dataset)
        print('INDIVIDUAL COMPLEXITY:', ind_complexity)
        print('INDIVIDUAL BIN', get_bin(ind_complexity, bin_width))

    print('INITIAL POP FITNESS HIST')
    for key, value in pop_hist_fitnesses.items():
        print(f"{key}: {len(value)}", end=' ')
    print()
    print('Bin width:', bin_width)

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
    best_val_fitness = best_test_fit_list[0] # Initialize best validation fitness

    print('BEST VAL FITNESS', best_val_fitness)
    # Save complexities
    iodc = [IODC(max_IODC, z, best_of_run, train_dataset)]
    print('INITIAL inv IODC:', iodc[0])
    # Save distributions
    target_histogram = [[target_hist[key] for key in sorted(target_hist.keys())]]
    #population_histogram = [[pop_hist_fitnesses[key] for key in sorted(pop_hist_fitnesses.keys())]]
    population_histogram = [get_population_len_histogram(pop_hist_fitnesses)]

    for gen in range(1, GENERATIONS + 1):  
        print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)
        print('NR GEN NO IMPROV:', nr_gen_no_improvement)

        # Reset population histogram
        pop_hist_fitnesses = reset_pop_hist(list(pop_hist_fitnesses.keys()))

        best_of_runs = {}

        new_pop = []
        new_train_fitnesses = []

        while len(new_pop) < POP_SIZE:
            # print('LEN NEW POP:', len(new_pop))
            
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
                
                parent_iodc = IODC(max_IODC, z, parent, train_dataset)
                parent2_iodc = IODC(max_IODC, z, parent2, train_dataset)

                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = get_bin(parent_iodc, bin_width),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f,
                                      best_of_runs = best_of_runs,
                                      individual = parent):

                    # Add parent to population and to histograms
                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, get_bin(parent_iodc, bin_width), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen

                if len(new_pop) < POP_SIZE and check_bin_capacity(target_hist, pop_hist_fitnesses, get_bin(parent2_iodc, bin_width),
                                                                  ind_fitness = parent2_fitness,
                                                                  best_of_run_f = best_of_run_f,
                                                                  best_of_runs = best_of_runs,
                                                                  individual = parent2):

                    new_pop.append(parent2)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, get_bin(parent2_iodc, bin_width), parent2_fitness)
                    new_train_fitnesses.append(parent2_fitness)

                    if parent2_fitness < best_of_run_f:
                        best_of_run_f = parent2_fitness
                        best_of_run = deepcopy(parent2)
                        best_of_run_gen = gen


            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent.mutation()

                parent_fitness = fitness(parent, train_dataset, train_target)
                parent_iodc = IODC(max_IODC, z, parent, train_dataset)

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')

                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = get_bin(parent_iodc, bin_width),
                                      ind_fitness = parent_fitness,
                                      best_of_run_f = best_of_run_f,
                                      best_of_runs = best_of_runs,
                                      individual = parent):

                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, get_bin(parent_iodc, bin_width), parent_fitness)
                    new_train_fitnesses.append(parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen
            
            # NOTE: Replication may also occur if no condition is met
            else:

                parent_fitness = fitness(parent, train_dataset, train_target)
                parent_iodc = IODC(max_IODC, z, parent, train_dataset)

                if check_bin_capacity(target_hist, pop_hist_fitnesses, ind_bin = get_bin(parent_iodc, bin_width),
                                      ind_fitness = fitness(parent, train_dataset, train_target),
                                      best_of_run_f = best_of_run_f,
                                      best_of_runs = best_of_runs,
                                      individual = parent):

                    new_pop.append(parent)
                    target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, get_bin(parent_iodc, bin_width), parent_fitness)

                    if parent_fitness < best_of_run_f:
                        best_of_run_f = parent_fitness
                        best_of_run = deepcopy(parent)
                        best_of_run_gen = gen

        # Check if true best of run exceeds histogram
        best_complexity = IODC(max_IODC, z, best_of_run, train_dataset)
        print('END OF GEN. COMPLEXITY OF BEST INDIVIDUAL:', best_complexity)
        print('BEST INDIVIDUAL BIN')
        print(get_bin(best_complexity, bin_width))

        # Actual best of run is not within histogram bounds AND we are still not exceeding the max iters with no val improvement
        if not all(best_of_run_f < ind_key[0] for ind_key in best_of_runs.keys()) and nr_gen_no_improvement < MAX_ITER_NO_IMPROV:
            print('BEST INDIVIDUAL WITHIN BOUNDS IS WORST THAN BEST INDIVIDUAL OUT OF BOUNDS')
            print('BEST OF RUN FITNESS', best_of_run_f)
            print('BEST OF RUNS DICT', best_of_runs)

            # Get best of run individual and respectuve bin
            best_fitness_key = min(best_of_runs, key=lambda x: x[0])
            print('BEST FITNESS OUT OF BOUND:', best_fitness_key)

            # Update best of run
            best_of_run_f = best_fitness_key[0]
            best_of_run = deepcopy(best_of_runs[best_fitness_key])

            print('NEW BEST OF RUN!', best_of_run)
            print('NEW BEST OF RUN FITNESS!', best_of_run_f)
            print('NEW BEST OF RUN COMPLEXITY!:', IODC(max_IODC, z, best_of_run, train_dataset))
            print('BIN WIDTH', bin_width)
            print('NEW BEST OF RUN BIN!:', get_bin(IODC(max_IODC, z, best_of_run, train_dataset), bin_width))
            
            # Find all other best of run individuals in a smaller bin then the best of run individual
            keys_with_smaller_second_element = [key for key in best_of_runs.keys() if key[1] < best_fitness_key[1]]
            keys_with_smaller_second_element.append(best_fitness_key)

            print('OTHER INDIVIDUALS IN A SMALLER BIN:', keys_with_smaller_second_element)

            to_add_individuals = [best_of_runs[key] for key in keys_with_smaller_second_element]
            print('TO ADD INDIVIDUALS', to_add_individuals)
            

            print('ADDING BEST INDIVIDUALS AND REMOVING WORST ONES')
            # Swap worst individual by best individual
            while len(to_add_individuals) > 0:
                # Worst individual index
                print('TRAIN FITNESSES', new_train_fitnesses)
                worst_idx = new_train_fitnesses.index(max(new_train_fitnesses))
                print('WORST INDIVIDUAL INDEX', worst_idx)

                # Get worst individual
                worst_ind = new_pop[worst_idx]

                # Remove him from population histogram
                worst_ind_bin = get_bin(IODC(max_IODC, z, worst_ind, train_dataset), bin_width)
                print('BIN OF WORST INDIVIDUAL:', worst_ind_bin)
                print('OLD POP HIST FITNESSES IN THAT BIN', pop_hist_fitnesses[worst_ind_bin])
                pop_hist_fitnesses[worst_ind_bin].remove(new_train_fitnesses[worst_idx])
                print('NEW POP HIST FITNESSES IN THAT BIN', pop_hist_fitnesses[worst_ind_bin])

                # Replace worst individual by the one to add
                new_ind_fitness = keys_with_smaller_second_element[0][0]
                new_ind_bin = keys_with_smaller_second_element[0][1]
                print('NEW BEST INDIVIDUAL FITNESS AND BIN:', new_ind_fitness, '|', new_ind_bin)

                new_pop[worst_idx] = to_add_individuals[0]
                new_train_fitnesses[worst_idx] = new_ind_fitness
                print('NEW TRAIN FITNESSES', new_train_fitnesses)

                to_add_individuals.pop(0)
                keys_with_smaller_second_element.pop(0)
                print('INDIVIDUAL REMOVED FROM:', to_add_individuals)
                print('INDIVIDUAL KEY REMOVED FROM:', keys_with_smaller_second_element)

                # Add new individual to histogram
                target_hist, pop_hist_fitnesses = update_hist(target_hist, pop_hist_fitnesses, new_ind_bin, new_ind_fitness)
                
                print('UPDATED TARGET HIST:', target_hist)
                print('UPDATED POP HIST:', pop_hist_fitnesses)


        print(best_of_run)
        new_best_complexity = IODC(max_IODC, z, best_of_run, train_dataset)
        print('BEST INDIVIDUAL COMPLEXITY', new_best_complexity)
        print('BEST INDIVIDUAL BIN')
        print(get_bin(new_best_complexity, bin_width))

        population = new_pop
        train_fitnesses = new_train_fitnesses

        # TODO: Should it be like this? Should i always normalize the fitnesses
        # to make the highest RMSE be the lowest fitness using the max train fitness in that generation?
        target_hist = update_target_hist(pop_hist_fitnesses, max(train_fitnesses))

        print('NEW POP HIST')
        for key, value in pop_hist_fitnesses.items():
            print(f"{key}: {len(value)}", end=' ')
        print()

        print('NEW TARGET HIST')
        print(target_hist)


        # train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]        
           
        # print("________________________")
        # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
        # best_of_run.print_tree()

        print('END OF GENERATION BEST OF RUN FITNESS:', best_of_run_f)
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        new_val_fitness = fitness(best_of_run, test_dataset, test_target)

        if new_val_fitness < best_val_fitness:
            best_val_fitness = new_val_fitness
            print('NEW BEST VAL FITNESS', best_val_fitness)
            nr_gen_no_improvement = 0
        else:
            nr_gen_no_improvement += 1
        
        best_test_fit_list.append(new_val_fitness)
        # Save complexity
        iodc.append(IODC(max_IODC, z, best_of_run, train_dataset))
        # Save distributions
        target_histogram.append([target_hist[key] for key in sorted(target_hist.keys())])
        #population_histogram.append([pop_hist_fitnesses[key] for key in sorted(pop_hist_fitnesses.keys())])
        population_histogram.append(get_population_len_histogram(pop_hist_fitnesses))

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, iodc, target_histogram, population_histogram
    
