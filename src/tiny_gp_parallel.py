from random import randint, random
from copy import deepcopy
import numpy as np
import time
from multiprocessing import Pool

from configs import *
from gptree import GPTree
from complexity_measures import IODC, polynomial_analysis
                   
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


    # print('PREDICTIONS')
    # individual.print_tree()
    # print(preds)
    
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



    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    print('----------------------------------------------------------- END ------------------------------------')
    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    best_test_fit_list = [fitness(best_of_run, test_dataset, test_target)]
    iodc = [IODC(best_of_run, train_dataset)]
    p_analysis = [polynomial_analysis(best_of_run)]
    overfit = [0]
    btp = best_test_fit_list[0]
    tbtp = best_of_run_f


    for gen in range(1, GENERATIONS + 1):  
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

                parent.crossover(parent2)

                # print('AFTER CROSSOVER')
                # print('PARENT1')
                # parent.print_tree()
                # print(parent.create_expression())
                # print(parent.tree_lambda.expr)
                # print('PARENT2')
                # parent2.print_tree()
                # print(parent2.create_expression())
                # print(parent2.tree_lambda.expr)


                parent.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!
                parent2.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

                # print('AFTER UPDATING LAMBDA FUNCTION')
                # print('PARENT1')
                # print(parent.create_expression())
                # print(parent.tree_lambda.expr)
                # print('PARENT2')
                # print(parent2.create_expression())
                # print(parent2.tree_lambda.expr)

                # print('-----------------')


                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')
                
                new_pop.append(parent)

                # if len(new_pop) < POP_SIZE:
                #     new_pop.append(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:
                # start = time.time()
                parent.mutation()
                
                # print('AFTER MUTATION')
                # parent.print_tree()
                # print(parent.create_expression())
                # print(parent.tree_lambda.expr)
                
                # parent.create_lambda_function() # CREATE LAMBDA FUNCTION HERE!

                # print('AFTER UPDATING EXPRESSION')
                # print(parent.create_expression())
                # print(parent.tree_lambda.expr)

                # print('--------------')

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')
                
                new_pop.append(parent)
            
            # NOTE: Replication may also occur if no condition is met
            else:
                new_pop.append(parent)
            
        population = new_pop

        # print('NEW POPULATION')
        # for ind in population:
        #     ind.print_tree()
        #     print(ind.create_expression())
        #     print(ind.tree_lambda.expr)
        #     print('----------------------')


        train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
        
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
        # TODO: Confirm!
        # Performance of the individual with the best train fitness in this generation
        test_performance = fitness(deepcopy(population[train_fitnesses.index(min(train_fitnesses))]), test_dataset, test_target)

        if min(train_fitnesses) > test_performance:
            overfit.append(0)
        else:
            if test_performance < btp:
                btp = test_performance
                overfit.append(0)
                tbtp = min(train_fitnesses)
            else:
                overfit.append(abs(min(train_fitnesses) - test_performance) - abs(tbtp - btp))
            
        # ---------------------------------------------------------------------------- #
           
        # print("________________________")
        # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
        # best_of_run.print_tree()
        
        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        best_test_fit_list.append(test_performance)
        iodc.append(IODC(best_of_run, train_dataset))
        p_analysis.append(polynomial_analysis(best_of_run))

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen, iodc, p_analysis
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
