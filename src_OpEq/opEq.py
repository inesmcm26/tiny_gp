import numpy as np

from configs_OpEq import *


#####################################################
#                  Target Histogram                 #
#####################################################

def init_target_hist(pop_hist_fitnesses, max_fitness):
    hist = {}

    nr_bins = max(pop_hist_fitnesses.keys())

    # nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))

    if TARGET == 'FLAT':

        bin_capacity = int(POP_SIZE / nr_bins)

        for i in range(1, nr_bins + 1):
            hist[i] = bin_capacity
    
    elif TARGET == 'DYN':

        mean_fitnesses = {bin: - np.mean(pop_hist_fitnesses[bin]) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}

        all_means_diff_zero = [mean_fit for mean_fit in mean_fitnesses.values() if mean_fit != 0]
        # Fitnesses are normalized for a minimization problem
        # all_fitnesses = {bin: np.mean(max_fitness - np.array(pop_hist_fitnesses[bin])) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}

        if min(mean_fitnesses.values()) < 0:
            for bin in mean_fitnesses:
                if mean_fitnesses[bin] != 0:
                    mean_fitnesses[bin] += (abs(max(all_means_diff_zero)) + abs(min(all_means_diff_zero)))
        
        for bin in mean_fitnesses.keys():
            hist[bin] = int(POP_SIZE * (mean_fitnesses[bin] / sum(mean_fitnesses.values())))
    
    return hist

def update_target_hist(pop_hist_fitnesses, max_fitness):

    hist = {}

    if TARGET == 'FLAT':
    
        nr_bins = max(pop_hist_fitnesses.keys())
    
        bin_capacity = int(POP_SIZE / nr_bins)

        for i in range(1, nr_bins + 1):
            hist[i] = bin_capacity
    
    elif TARGET == 'DYN':

        mean_fitnesses = {bin: - np.mean(pop_hist_fitnesses[bin]) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}

        all_means_diff_zero = [mean_fit for mean_fit in mean_fitnesses.values() if mean_fit != 0]
        # Fitnesses are normalized for a minimization problem
        # all_fitnesses = {bin: np.mean(max_fitness - np.array(pop_hist_fitnesses[bin])) if len(pop_hist_fitnesses[bin]) > 0 else 0 for bin in pop_hist_fitnesses.keys()}
            
        if min(mean_fitnesses.values()) < 0:
            for bin in mean_fitnesses:
                if mean_fitnesses[bin] != 0:
                    mean_fitnesses[bin] += (abs(max(all_means_diff_zero)) + abs(min(all_means_diff_zero)))
        
        for bin in mean_fitnesses.keys():
            hist[bin] = int(POP_SIZE * (mean_fitnesses[bin] / sum(mean_fitnesses.values())))

    return hist


#####################################################
#                Population Histogram               #
#####################################################

def reset_pop_hist(bins):
    """
    Reset population histigram fo fitnesses to empty lists
    """

    return {bin: [] for bin in bins}

def init_hist(population, train_fitnesses):
    """
    Calculates the list of fitnesses of the individuals belonging to each bin
    """

    sizes = [ind.size() for ind in population]

    print('SIZES:', sizes)

    biggest_ind = population[sizes.index(max(sizes))]

    nr_bins = biggest_ind.get_bin()

    # nr_bins = int((HIST_INITIAL_LIMIT / BIN_WIDTH))

    pop_hist_fitness = {}

    # For each individual, find its bin
    for idx, ind in enumerate(population):
        ind_bin = ind.get_bin()

        # Add the individual fitness to the bin
        if ind_bin in pop_hist_fitness.keys():
            pop_hist_fitness[ind_bin].append(train_fitnesses[idx])
        else:
            pop_hist_fitness[ind_bin] = [train_fitnesses[idx]]

    # For bins with no individuals, initialize an empty list
    for i in range(1, nr_bins + 1):
        if i not in pop_hist_fitness.keys():
            pop_hist_fitness[i] = []

    return pop_hist_fitness

def check_bin_capacity(target_hist, pop_fitness_hist, ind_bin, ind_fitness, best_of_run_f):
    """
    Check if individual can be added to the population given the ideal target distribution
    """

    # print('POP HIST AVAILABILITY')
    # for key, value in pop_fitness_hist.items():
    #     print(f"{key}: {len(value)}", end=' ')
    # print()

    # print('IND BIN', ind_bin)
    # print('TARGET CAPACITY', target_hist[ind_bin])
    # print('NR OF INDS IN BIN', len(pop_fitness_hist[ind_bin]))


    # If in range
    if ind_bin in pop_fitness_hist.keys():
        # Bin not full
        if len(pop_fitness_hist[ind_bin]) < target_hist[ind_bin]:
            # print('NOT FULL', len(pop_fitness_hist[ind_bin]), '<', target_hist[ind_bin])
            return True
        # Full bin but best of bin -> exceed capacity
        elif len(pop_fitness_hist[ind_bin]) >= target_hist[ind_bin]:
            # print('BINS BEST FITNESS', min(pop_fitness_hist[ind_bin]) if len(pop_fitness_hist[ind_bin]) != 0 else 0)
            if len(pop_fitness_hist[ind_bin]) == 0:
                return True
            elif ind_fitness < min(pop_fitness_hist[ind_bin]):
                return True
        
        # print('IND FITNESS:', ind_fitness)
        # print('IND BIN:', ind_bin)
        # print('BIN OCUPANCY:', len(pop_fitness_hist[ind_bin]))
        # print('BIN CAPACITY:', target_hist[ind_bin])
            # print('FULL BUT BEST OF RUN', ind_fitness, '<', best_of_run_f)
    # Out of range but best of run -> add new bin
    elif ind_fitness < best_of_run_f:
        # print('BEST OF RUN FITNESS:', best_of_run_f)
        # print('NEW INDIVIDUAL FITNESS:', ind_fitness)
        # print('NEW IND BIN:', ind_bin)
        # print('IND FITNESS:', ind_fitness)
        # print('BEST OF RUN FITNESS', best_of_run_f)
        # print('OUT OF RANGE BUT BEST OF RUN')
        # print(ind_fitness, '<', best_of_run_f)
        return True
    # print('REJECTING')
    return False

def update_hist(target_hist, pop_hist_fitness, ind_bin, ind_fitness):
    """
    When individual is added to the population, update the population histogram
    and maybe the target population when the new bine xceed the old upper bound
    """

    # Existing bin
    if ind_bin in pop_hist_fitness.keys():
        pop_hist_fitness[ind_bin].append(ind_fitness)

        # print('NEW POP HIST AVAILABILITY')
        # for key, value in pop_hist_fitness.items():
        #     print(f"{key}: {len(value)}", end=' ')
        # print()
    
    # New bin
    else:
        # print('ADD NEW BINS UNTIL', ind_bin)
        # Add new bins
        for new_bin in range(max(target_hist.keys()) + 1, ind_bin + 1):
            target_hist[new_bin] = 0
            pop_hist_fitness[new_bin] = []

        pop_hist_fitness[ind_bin].append(ind_fitness)

        # print('NEW POP HIST AVAILABILITY')
        # for key, value in pop_hist_fitness.items():
        #     print(f"{key}: {len(value)}", end=' ')
        # print()
        
        # print('NEW TARGET HIST', target_hist)
    
    return target_hist, pop_hist_fitness

def get_population_len_histogram(population_fitness_histogram):
    pop_len_hist = []
    
    for ind_bin in sorted(population_fitness_histogram.keys()):
        pop_len_hist.append(len(population_fitness_histogram[ind_bin]))
    
    return pop_len_hist

def get_best_ind_in_bins(population_fitness_histogram):
    best_fits = {}

    for ind_bin in sorted(population_fitness_histogram.keys()):
        if len(population_fitness_histogram[ind_bin]) > 0:
            best_fits[ind_bin] = min(population_fitness_histogram[ind_bin])
        else:
            best_fits[ind_bin] = None

    return best_fits