from random import randint, random
from copy import deepcopy
import numpy as np
import time
from multiprocessing import Pool

from configs import *
from gptree import GPTree
from complexity_measures import init_IODC, IODC, mean_IODC, polynomial_analysis, mean_polynomial_analysis
from complexity_measures import init_slope_based_complexity, slope_based_complexity, mean_slope_based_complexity
                   
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

    # print('TRAIN DATASET')
    # print(train_dataset)

    population = init_population(terminals) 

    # for ind in population:
    #     ind.print_tree()
    #     print(ind.create_expression())
    #     print(ind.tree_lambda.expr)
    #     print('----------------------')

    # Upper bounds for complexities
    z, max_IODC = init_IODC(train_dataset, train_target)
    max_slope_complexity, p_js, q_js = init_slope_based_complexity(train_dataset, train_target)

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
    iodc = [IODC(max_IODC, z, best_of_run, train_dataset)]
    p_analysis = [polynomial_analysis(best_of_run)]
    slope = [slope_based_complexity(max_slope_complexity, p_js, q_js, best_of_run, train_dataset)]
    # Save mean complexities
    mean_iodc = [mean_IODC(max_IODC, z, population, train_dataset)]
    mean_p_analysis = [mean_polynomial_analysis(population)]
    mean_slope = [mean_slope_based_complexity(max_slope_complexity, p_js, q_js, population, train_dataset)]
    # Save overfitting
    overfit = [0]
    btp = best_test_fit_list[0]
    tbtp = best_of_run_f

    for gen in range(1, GENERATIONS + 1):  
        # print('------------------------------------------ NEW GEN ------------------------------------------')
        print(gen)
        start = time.time()

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
                
                new_pop.append(parent)

                if len(new_pop) < POP_SIZE:
                    new_pop.append(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:

                parent.mutation()

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')
                
                new_pop.append(parent)
            
            # NOTE: Replication may also occur if no condition is met
            else:
                new_pop.append(parent)
            
        population = new_pop

        print('NEW POP DONE', time.time() - start)

        start = time.time()

        train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
        print('NEW TRAIN FITNESSES DONE', time.time() - start)
        
        start = time.time()
        test_fitnesses = [fitness(ind, test_dataset, test_target) for ind in population]
        print('NEW TEST FITNESSES DONE', time.time() - start)

        # # You can set the number of processes as desired
        # num_processes = 8

        # pool = Pool(processes=num_processes)

        # train_fitnesses = pool.starmap(fitness, [(ind, train_dataset, train_target) for ind in population])

        # pool.close()
        # pool.join()
        
        if min(train_fitnesses) < best_of_run_f:
            best_of_run_f = min(train_fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])        

        # ---------------------------------- Overfit ---------------------------------- #
        # # TODO: Confirm!
        # # Performance of the individual with the best train fitness in this generation
        # test_performance = fitness(deepcopy(population[train_fitnesses.index(min(train_fitnesses))]), test_dataset, test_target)

        # if min(train_fitnesses) > test_performance:
        #     overfit.append(0)
        # else:
        #     if test_performance < btp:
        #         btp = test_performance
        #         overfit.append(0)
        #         tbtp = min(train_fitnesses)
        #     else:
        #         overfit.append(abs(min(train_fitnesses) - test_performance) - abs(tbtp - btp))
            
        # ---------------------------------------------------------------------------- #
           
        # print("________________________")
        # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
        # best_of_run.print_tree()
        
        # Save best train performance
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        # Save test performance
        best_test_fit_list.append(fitness(best_of_run, test_dataset, test_target))

        # Save mean train performance
        mean_train_fit_list.append(np.mean(train_fitnesses))
        mean_test_fit_list.append(np.mean(test_fitnesses))

        # Save complexities
        start = time.time()
        iodc.append(IODC(max_IODC, z, best_of_run, train_dataset))
        p_analysis.append(polynomial_analysis(best_of_run))
        slope.append(slope_based_complexity(max_slope_complexity, p_js, q_js, best_of_run, train_dataset))
        print('BEST COMPLEXITIES DONE', time.time() - start)

        # Save mean complexities
        start = time.time()
        mean_iodc.append(mean_IODC(max_IODC, z, population, train_dataset))
        print('MEAN IODC DONE', time.time() - start)

        start = time.time()
        mean_p_analysis.append(mean_polynomial_analysis(population))
        print('MEAN POLYNOMIAL DONE', time.time() - start)

        start = time.time()
        mean_slope.append(mean_slope_based_complexity(max_slope_complexity, p_js, q_js, population, train_dataset))
        print('MEAN SLOPE DONE', time.time() - start)

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, mean_train_fit_list, mean_test_fit_list, \
        iodc, p_analysis, slope, mean_iodc, mean_p_analysis, mean_slope
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
