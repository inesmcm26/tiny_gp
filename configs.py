POP_SIZE        = 12   # population size
MIN_DEPTH       = 1    # minimal initial random tree depth
MAX_DEPTH       = 3    # maximal initial random tree depth
GENERATIONS     = 3  # maximal number of generations to run evolution
TOURNAMENT_SIZE = 3    # size of tournament for tournament selection
XO_RATE         = 0.4  # crossover rate 
PROB_MUTATION   = 0.5  # per-node mutation probability
FITNESS         = 'RMSE'

TERMINALS = [f'x{i}' for i in range(2)]