# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import randint, random
from statistics import mean
from copy import deepcopy
import numpy as np

from configs import *
from gptree import GPTree
                   
def init_population(terminals):
    """
    Ramped half-and-half initialization
    """

    # Number of individuals of each depth and initialized with each method
    inds_per_depth = int((POP_SIZE / MAX_DEPTH) / 2)

    # print('POP SIZE', POP_SIZE)
    # print('MAX DEPTH', MAX_DEPTH)
    # print('INDS PER DEPTH AND METHOD', inds_per_depth)

    pop = []
    for max_depth in range(MIN_DEPTH, MAX_DEPTH + 1):
        
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
        max_depth = randint(MIN_DEPTH, MAX_DEPTH)
        grow = True if random() < 0.5 else False
        ind = GPTree(terminals = terminals)
        ind.random_tree(grow = grow, max_depth = max_depth)
        pop.append(ind) 

    return pop

def fitness(individual, dataset, target):

    # Calculate predictions
    preds = [individual.compute_tree(obs) for obs in dataset]
    
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

    # print('POP SIZE', POP_SIZE)
    # print('LEN POP', len(population))
    
    # print('POPULATION:')
    # for ind in population:
    #     print('IND')
    #     ind.print_tree()
        
        # print('ALG EXPR')
        # print(ind.create_expression())


        # print('PREDICTIONS')
        # print([ind.compute_tree(obs) for obs in train_dataset])
        # print('TARGET', target)
        # print('FITNESS:', fitness(ind, train_dataset, target))

    train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]

    best_of_run_f = min(train_fitnesses)
    best_of_run_gen = 0
    best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

    # # CROSSOVER TEST
    # parent = tournament(population, train_fitnesses)
    # parent2 = tournament(population, train_fitnesses)
    # parent.crossover(parent2)

    # # MUTATION TEST
    # parent = tournament(population, train_fitnesses)
    # parent.mutation()

    best_train_fit_list = [best_of_run_f]
    best_ind_list = [best_of_run.create_expression()]
    best_test_fit_list = [fitness(best_of_run, test_dataset, test_target)]


    for gen in range(1, GENERATIONS + 1):  

        new_pop=[]

        while len(new_pop) < POP_SIZE:
            
            prob = random()
            # print('PROB', prob)

            parent = tournament(population, train_fitnesses)

            # Crossover
            if prob < XO_RATE:
                parent2 = tournament(population, train_fitnesses)

                parent.crossover(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:
                parent.mutation()

            # NOTE: Replication may also occur if no condition is met

            new_pop.append(parent)
            
        population = new_pop

        train_fitnesses = [fitness(ind, train_dataset, train_target) for ind in population]
        
        if min(train_fitnesses) < best_of_run_f:
            best_of_run_f = min(train_fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[train_fitnesses.index(min(train_fitnesses))])

            # print("________________________")
            # print("gen:", gen, ", best_of_run_f:", round(min(train_fitnesses), 3), ", best_of_run:") 
            # best_of_run.print_tree()
        

        best_train_fit_list.append(best_of_run_f)
        best_ind_list.append(best_of_run.create_expression())
        best_test_fit_list.append(fitness(best_of_run, test_dataset, test_target))

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f, 3)))
    # best_of_run.print_tree()

    return best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen
    
# if __name__== "__main__":
#   best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen = evolve()

#   print(best_train_fit_list, best_test_fit_list, best_ind_list, best_of_run_gen)
