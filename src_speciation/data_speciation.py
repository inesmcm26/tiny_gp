from random import randint, seed
import pandas as pd

NR_FEATS = 2
TRAIN_PERC = 0.7

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

    train_dataset = dataset[:int(TRAIN_PERC*len(dataset))]
    train_target = target[:int(TRAIN_PERC*len(target))]

    test_dataset = dataset[int(TRAIN_PERC*len(dataset)):]
    test_target = target[int(TRAIN_PERC*len(target)):]

    return train_dataset, test_dataset, train_target, test_target

def read_dataset(name, run_nr):
    PATH = '/home/ines/Documents/tese/tiny_gp/data/' + name

    train = pd.read_csv(PATH + f'/train_{run_nr}.csv', index_col = 0)
    test = pd.read_csv(PATH + f'/test_{run_nr}.csv', index_col = 0)
    
    train_dataset = train.drop('Target', axis = 1).to_numpy()
    train_target = train['Target'].to_numpy()

    test_dataset = test.drop('Target', axis = 1).to_numpy()
    test_target = test['Target'].to_numpy()

    return train_dataset, test_dataset, train_target, test_target
