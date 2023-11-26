import numpy as np

from src_OpEq_complexity.configs_OpEq_complexity import *
from src_OpEq_complexity.complexity_measures import IODC


def get_bin(complexity, bin_width):
    return np.ceil(complexity / bin_width)


#####################################################
#                  Target Histogram                 #
#####################################################

def init_target_hist(pop_hist_fitnesses, max_fitness):
    hist = {}

    nr_bins = int(len(pop_hist_fitnesses.keys()))

    if TARGET == 'FLAT':

        bin_capacity = int(POP_SIZE / nr_bins)


        for i in range(1, nr_bins + 1):
            hist[i] = bin_capacity
    
    elif TARGET == 'DYN':

        # Fitnesses are normalized for a minimization problem
        all_fitnesses = {bin: np.mean(max_fitness - np.array(pop_hist_fitnesses[bin])) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}
        
        for bin in range(1, nr_bins + 1):
            hist[bin] = int(np.round(POP_SIZE *  (all_fitnesses[bin] / sum(all_fitnesses.values()))))

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


#####################################################
#                Population Histogram               #
#####################################################

def reset_pop_hist(bins):
    """
    Reset population histigram fo fitnesses to empty lists
    """

    return {bin: [] for bin in bins}

def init_hist(population, train_fitnesses, max_IODC, z, dataset):
    """
    Calculates the list of fitnesses of the individuals belonging to each bin.

    Population and train_fitnesses are equally ordered
    """

    pop_hist_fitness = {}

    
    inds_complexities = [IODC(max_IODC, z, ind, dataset) for ind in population]

    # Complexities range from 0 to infinity
    max_comp = max(inds_complexities)

    # Let's define the bin width in relation to the maximum initial complexity -> TODO: make this smarter!
    bin_width = max_comp / INITIAL_NR_BINS

    # For each individual, calculate its complexity
    for idx in range(len(population)):
        ind_bin = get_bin(inds_complexities[idx], bin_width)

        # Add the individual fitness to the bin
        if ind_bin in pop_hist_fitness.keys():
            pop_hist_fitness[ind_bin].append(train_fitnesses[idx])
        else:
            pop_hist_fitness[ind_bin] = [train_fitnesses[idx]]

    # For bins with no individuals, initialize an empty list
    for i in range(1, INITIAL_NR_BINS + 1):
        if i not in pop_hist_fitness.keys():
            pop_hist_fitness[i] = []

    return pop_hist_fitness, bin_width

def check_bin_capacity(target_hist, pop_fitness_hist, ind_bin, ind_fitness, best_of_run_f, nr_iter_no_improv):
    """
    Check if individual can be added to the population given the ideal target distribution
    """

    # print('POP HIST AVAILABILITY')
    # for key, value in pop_fitness_hist.items():
    #     print(f"{key}: {len(value)}", end=' ')
    # print()

    print('-> CHECK BIN CAPACITY')
    print('IND BIN', ind_bin)

    # If in range
    if ind_bin in pop_fitness_hist.keys():
        # Bin not full
        if len(pop_fitness_hist[ind_bin]) < target_hist[ind_bin]:
            print('NOT FULL', len(pop_fitness_hist[ind_bin]), '<', target_hist[ind_bin])
            return True
        # Full bin but best of run -> exceed capacity
        elif len(pop_fitness_hist[ind_bin]) >= target_hist[ind_bin] and ind_fitness < best_of_run_f:
            print('FULL BUT BEST OF RUN', ind_fitness, '<', best_of_run_f)
            return True
    # Out of range but still under max number iterations with no improvement -> add new bin
    elif nr_iter_no_improv < MAX_ITER_NO_IMPROV:
        print('OUT OF RANGE. NR ITER NO IMPROV:', nr_iter_no_improv)
        return True
    
    return False

def update_hist(target_hist, pop_hist_fitness, ind_bin, ind_fitness):
    """
    When individual is added to the population, update the population histogram
    and maybe the target population when the new bin exceeds the old upper bound
    """

    # Existing bin
    if ind_bin in pop_hist_fitness.keys():
        print('ADD NEW IND to bin', ind_bin)
        pop_hist_fitness[ind_bin].append(ind_fitness)

        print('NEW POP HIST AVAILABILITY')
        for key, value in pop_hist_fitness.items():
            print(f"{key}: {len(value)}", end=' ')
        print()
    
    # New bin
    else:
        print('ADD NEW BINS UNTIL', ind_bin)
        # Add new bins
        for new_bin in range(max(target_hist.keys()) + 1, ind_bin + 1):
            target_hist[new_bin] = 1
            pop_hist_fitness[new_bin] = []

        pop_hist_fitness[ind_bin].append(ind_fitness)

        print('NEW POP HIST AVAILABILITY')
        for key, value in pop_hist_fitness.items():
            print(f"{key}: {len(value)}", end=' ')
        print()
        
        print('NEW TARGET HIST', target_hist)
    
    return target_hist, pop_hist_fitness

def get_population_len_histogram(population_fitness_histogram):
    pop_len_hist = []
    
    for ind_bin in sorted(population_fitness_histogram.keys()):
        pop_len_hist.append(len(population_fitness_histogram[ind_bin]))
    
    return pop_len_hist