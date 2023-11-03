# tiny genetic programming by © moshe sipper, www.moshesipper.com
from random import randint, seed, random
from statistics import mean
from copy import deepcopy
import numpy as np

from configs import *
from data import generate_dataset
from gptree import GPTree
                   
def init_population():
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
            ind = GPTree()
            ind.random_tree(grow = True, max_depth = max_depth)

            print('GROW')
            print('IND')
            ind.print_tree()

            pop.append(ind) 
        
        # Full
        for _ in range(inds_per_depth):
            ind = GPTree()
            ind.random_tree(grow = False, max_depth = max_depth)   

            print('FULL')
            print('IND')
            ind.print_tree()

            pop.append(ind) 


    # Edge case
    while len(pop) != POP_SIZE:
        # Generate random tree with random method to fill population
        max_depth = randint(MIN_DEPTH, MAX_DEPTH)
        grow = True if random() < 0.5 else False
        ind = GPTree()
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

    print('INDEXES OF INDIVIDUALS TO GO TO TORNAMENT', tournament)

    # Get their fitness values
    tournament_fitnesses = [fitnesses[tournament[i]] for i in range(TOURNAMENT_SIZE)]

    print('IND FITNESSES', tournament_fitnesses)
    
    # Return the winner
    return deepcopy(population[tournament[tournament_fitnesses.index(min(tournament_fitnesses))]]) 
            
def evolve():      
    seed()
    # TODO: get real dataset
    dataset, target= generate_dataset()

    # print('DATASET')
    # print(dataset)
    # print('TARGET')
    # print(target)

    population = init_population() 

    # print('POP SIZE', POP_SIZE)
    # print('LEN POP', len(population))

    for ind in population:
        print('IND')
        ind.print_tree()

        # print('PREDICTIONS')
        # print([ind.compute_tree(obs) for obs in dataset])
        # print('TARGET', target)
        print('FITNESS:', fitness(ind, dataset, target))

    # return


    best_of_run = None # Best of run individual
    best_of_run_f = 0 # Best of run fitness
    best_of_run_gen = 0 # Generation with best of run

    fitnesses = [fitness(ind, dataset, target) for ind in population]

    # CROSSOVER TEST
    parent = tournament(population, fitnesses)
    parent2 = tournament(population, fitnesses)
    parent.crossover(parent2)

    # # MUTATION TEST
    # parent = tournament(population, fitnesses)
    # parent.mutation()

    return

    for gen in range(GENERATIONS):  

        new_pop=[]

        while len(new_pop) < POP_SIZE:
            
            prob = random()
            print('PROB', prob)

            parent = tournament(population, fitnesses)

            print('TOURNAMENT WINNER')
            parent.print_tree()

            # Crossover
            if prob < XO_RATE:
                parent2 = tournament(population, fitnesses)

                print('TOURNAMENT WINNER')
                parent2.print_tree()

                parent.crossover(parent2)

            # Mutation
            elif prob < XO_RATE + PROB_MUTATION:
                parent.mutation()

            # NOTE: Replication may also occur if no condition is met

            new_pop.append(parent)
            
        population = new_pop

        fitnesses = [fitness(ind, dataset, target) for ind in population]

        print('NEW INDIVIDUALS FITNESSES', fitnesses)
        
        if max(fitnesses) > best_of_run_f:
            best_of_run_f = max(fitnesses)
            best_of_run_gen = gen
            best_of_run = deepcopy(population[fitnesses.index(max(fitnesses))])

            print("________________________")
            print("gen:", gen, ", best_of_run_f:", round(max(fitnesses),3), ", best_of_run:") 
            best_of_run.print_tree()

        # Optimal solution found
        if best_of_run_f == 0:
            break   
    
    print("\n\n_________________________________________________\nEND OF RUN\nbest_of_run attained at gen " + str(best_of_run_gen) +\
          " and has f=" + str(round(best_of_run_f,3)))
    best_of_run.print_tree()
    
if __name__== "__main__":
  evolve()
