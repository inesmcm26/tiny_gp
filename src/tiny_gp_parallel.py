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

    # print('POP SIZE', POP_SIZE)
    # print('MAX DEPTH', MAX_INITIAL_DEPTH)
    # print('INDS PER DEPTH AND METHOD', inds_per_depth)

    pop = []
    for max_depth in range(MIN_DEPTH, MAX_INITIAL_DEPTH + 1):
        
        # Grow
        for _ in range(inds_per_depth):
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = True, max_depth = max_depth)

            # print('GROW')
            # print('IND')
            # ind.print_tree()

            pop.append(ind) 
        
        # Full
        for _ in range(inds_per_depth):
            ind = GPTree(terminals = terminals)
            ind.random_tree(grow = False, max_depth = max_depth)   

            # print('FULL')
            # print('IND')
            # ind.print_tree()

            pop.append(ind) 


    # Edge case
    while len(pop) != POP_SIZE:
        # Generate random tree with random method to fill population
        max_depth = randint(MIN_DEPTH, MAX_INITIAL_DEPTH)
        grow = True if random() < 0.5 else False
        ind = GPTree(terminals = terminals)
        ind.random_tree(grow = grow, max_depth = max_depth)
        pop.append(ind) 

    return pop

def fitness(individual, dataset, target):

    start = time.time()
    # Calculate predictions
    preds = [individual.compute_tree(obs) for obs in dataset]
    # print('PREDICTIONS TIME', time.time() - start)
    
    if FITNESS == 'RMSE':
        return np.sqrt(np.mean((np.array(preds) - np.array(target)) ** 2))
    
    elif FITNESS == 'MAE':
        return np.mean(abs(np.array(preds) - np.array(target)))
    
    #  # inverse mean absolute error over dataset normalized to [0,1]
    # return 1 / (1 + mean([abs(individual.compute_tree(ds[0]) - ds[1]) for ds in dataset]))
                
def tournament(population, fitnesses):
    """
    Tournament selection
    """
    # Select random individuals to compete
    tournament = [randint(0, len(population)-1) for _ in range(TOURNAMENT_SIZE)]

    # print('INDEXES OF INDIVIDUALS TO GO TO TORNAMENT', tournament)

    # Get their fitness values
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]

    # print('IND FITNESSES', tournament_fitnesses)
    
    # Return the winner
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 
            
def evolve(train_dataset, test_dataset, train_target, test_target, terminals):

    # print('DATASET')
    # print(dataset)
    # print('TARGET')
    # print(target)

    population = init_population(terminals) 

    # ind = population[10]

    # ind.print_tree()

    # max_depth = randint(1, ind.depth())

    # print('DEPTH FORAAAAA', max_depth)

    # print('INDEXES FORAAAAA', ind.get_nodes_idx_above_depth(max_depth = max_depth))

    # return

    # print('POP SIZE', POP_SIZE)
    # print('LEN POP', len(population))
    
    # print('POPULATION:')
    # for ind in population:
        # print('IND')
        # ind.print_tree()
        
        # print('ALG EXPR')
        # print(ind.create_expression())


        # print('PREDICTIONS')
        # print([ind.compute_tree(obs) for obs in train_dataset])
        # print('TARGET', target)
        # print('FITNESS:', fitness(ind, train_dataset, target))
    # start = time.time()
    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
    # print('INITIAL FITNESS EVALUATION TIME', time.time() - start)

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

        # start = time.time()
        while len(new_pop) < POP_SIZE:
            
            prob = random()
            # print('PROB', prob)

            # start = time.time()
            parent = tournament(population, train_fitnesses)
            # print('TOURNAMENT TIME', time.time() - start)


            # Crossover
            if prob < XO_RATE:
                # print('CROSSOVER')
                # start = time.time()
                parent2 = tournament(population, train_fitnesses)
                # print('SECOND TOURNAMENT TIME', time.time() - start)

                # start = time.time()
                parent.crossover(parent2)
                # print('CROSSOVER TIME', time.time() - start)
                # print('------')

                if parent.depth() > MAX_DEPTH or parent2.depth() > MAX_DEPTH:
                    raise Exception('Crossover generated an individual that exceeds depth.')

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:
                # start = time.time()
                parent.mutation()
                # print('MUTATION TIME', time.time() - start)

                # parent.print_tree()

                # print('DEPTH AFTER MUTATION', parent.depth())

                if parent.depth() > MAX_DEPTH:
                    raise Exception('Mutation generated an individual that exceeds depth.')

            # NOTE: Replication may also occur if no condition is met

            new_pop.append(parent)
            
        population = new_pop
        # print('FILLED POPULATION TIME', time.time() - start)

        # You can set the number of processes as desired
        num_processes = 8

        pool = Pool(processes=num_processes)

        # start = time.time()
        train_fitnesses = pool.starmap(fitness, [(ind, train_dataset, train_target) for ind in population])
        # train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
        # print('END OF GEN FITNESS EVALUATION', time.time() - start)

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
