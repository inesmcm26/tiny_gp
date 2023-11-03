from random import randint, seed

NR_FEATS = 2

seed(0)

def target_func(obs): # evolution's target
    # x1 + x2 + x1*x2
    return obs[0] + obs[1] + obs[0]*obs[1]

def generate_dataset(): # generate 101 data points from target_func
    dataset = []
    target = []

    # Generate 20 random observations and corresponding targets
    for _ in range(10):
        obs = [randint(1, 10) for _ in range(NR_FEATS)]
        dataset.append(obs)
        target.append(target_func(obs))

    return dataset, target