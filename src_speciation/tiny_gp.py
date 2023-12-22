from random import randint, random, choice
from copy import deepcopy
import numpy as np
from sklearn.model_selection import KFold

from configs_speciation import *
from gptree import GPTree
from complexity_measures import init_IODC, IODC
from complexity_measures import init_slope_based_complexity, slope_based_complexity
                   
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
            
def evolve_species(population, train_dataset, train_target):

    # print('TRAIN DATASET')
    # print(train_dataset)

    # population = init_population(terminals) 

    # for ind in population:
    #     ind.print_tree()
    #     print(ind.number_operations())
    #     print(len(set(ind.used_features())))
    #     print('----------------------')

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    for gen in range(1, GENERATIONS_PER_SPLIT + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)

        new_pop=[]

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

        train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in new_pop]

        # Check if len population exceeded
        if len(new_pop) > POP_SIZE:
            print('POP SIZE EXCEEDED')
            # Remove worst individual
            idx_worst = train_fitnesses.index(max(train_fitnesses))
            new_pop.pop(idx_worst)
            train_fitnesses.pop(idx_worst)
        
        if len(new_pop) != POP_SIZE:
            raise Exception('POP SIZE EXCEEDED!!!')
            
        population = new_pop.copy()
        

    return population, train_fitnesses

def evolve(df_train, df_test, train_target, test_target, terminals):

    # Upper bounds for complexities
    z, max_IODC = init_IODC(df_train, train_target)
    max_slope_complexity = init_slope_based_complexity(df_train, train_target)

    top_train_fitnesses = []
    top_test_fitnesses = []

    best_of_run_f = 999999999999999999

    # Save best train performance
    best_train_fit_list = []
    best_ind_list = []
    # Save test performance
    best_test_fit_list = []
    # Save mean train performance
    mean_train_fit_list = []
    mean_test_fit_list = []
    # Save complexities
    iodc = []
    slope = []
    # Save complexity distributions
    iodc_distribution = []
    slope_distribution = []
    # Save mean complexities
    mean_iodc = []
    mean_slope = []
    # Save best ind size
    size = []
    # Save sizes
    size_distribution = []
    # Save mean sizes
    mean_size = []
    # Save measure of interpretability
    # Number of ops
    no = []
    no_distribution = []
    mean_no = []
    # Number of features used
    num_feats = []
    num_feats_distribution = []
    mean_number_feats = []

    for gen in range(1, GENERATIONS + 1):

        kf = KFold(n_splits = DATASET_SPLITS)

        if gen == 1:
            population = init_population(terminals) 
            print('FIRST GEN. NEW RANDOM POPULATION')
        else:
            population = total_pop
            print(f'INTERMEDIATE GEN {gen}. POP IS POP POPULATION')

        total_pop = []

        # Evolve a population for each split for GENERATIONS_PER_SPLIT generations
        for train_index, val_index in kf.split(df_train):

            X_train, X_val = df_train[train_index], df_train[val_index]
            y_train, y_val = train_target[train_index], train_target[val_index]

            species_pop, train_fitnesses = evolve_species(population, X_train, y_train)

            # print('SPECIES POP:', species_pop)

            # Sort fitnesses
            sorted_fitnesses = np.argsort(train_fitnesses)
            # Get POP_SIZE / DATASET_SPLITS individuals with best fitness
            # Get their train and test fitness
            num_inds_to_keep = int(POP_SIZE / DATASET_SPLITS)
            top_individuals = np.array(species_pop)[sorted_fitnesses][:num_inds_to_keep]

            # Add best individuals from each species
            total_pop.extend(top_individuals)

            # print('BEST INDIVIDUALS OF SPECIES FITNESS: ', train_fitnesses)
            # print('NEW TOTAL POP:', total_pop)

        # Get train fitnesses on entire train dataset of top species individuals
        top_train_fitnesses = [fitness(ind, df_train, train_target) for ind in total_pop]
        top_test_fitnesses = [fitness(ind, df_test, test_target) for ind in total_pop]

        if min(top_train_fitnesses) < best_of_run_f:
            best_of_run_f = min(top_train_fitnesses)
            best_of_run = deepcopy(total_pop[top_train_fitnesses.index(best_of_run_f)])  
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        best_test_fit_list.append(fitness(best_of_run, df_test, test_target))

        # Save mean train performance
        mean_train_fit_list.append(np.mean(top_train_fitnesses))
        mean_test_fit_list.append(np.mean(top_test_fitnesses))

        # Save complexities
        iodc.append(IODC(max_IODC, z, best_of_run, df_train))
        slope.append(slope_based_complexity(max_slope_complexity, best_of_run, df_train))

        # Save complexity distributions
        iodc_distribution.append([IODC(max_IODC, z, ind, df_train) for ind in population])
        slope_distribution.append([slope_based_complexity(max_slope_complexity, ind, df_train) for ind in population])

        # Save mean complexities
        mean_iodc.append(np.mean(iodc_distribution[-1]))
        mean_slope.append(np.mean(slope_distribution[-1]))

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

        print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
            " and has f=" + str(round(best_of_run_f, 3)))

    return best_train_fit_list, best_test_fit_list, best_ind_list, \
        mean_train_fit_list, mean_test_fit_list, \
        iodc, mean_iodc, iodc_distribution, \
        slope, mean_slope, slope_distribution, \
        size, mean_size, size_distribution, \
        no, mean_no, no_distribution


# Variations:
# - Train on the test splits instead, to increase speciation
# - Choose top individuals based on the same set it was trained on and ignore the rest -> Evaluate on the test set in the end DONE
# - Choose top individuals based on the entire train + val dataset to choose best individuals -> Evaluate on the test set in the end
# - Choose top individuals based on the set in which it was not trained on to choose best individuals -> Evaluate on the test set in the end 