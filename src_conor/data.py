from random import randint, uniform
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

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

    train_dataset = train.drop('Target', axis = 1)
    train_target = train['Target'].to_numpy()

    test_dataset = test.drop('Target', axis = 1)
    test_target = test['Target'].to_numpy()

    train_dataset, test_dataset = scale_numerical_features(train_dataset, test_dataset)

    return train_dataset, test_dataset, train_target, test_target

def scale_numerical_features(train_df, test_df):
    scaler = MinMaxScaler()

    train_df = scaler.fit_transform(train_df)

    test_df = scaler.transform(test_df)

    return train_df, test_df

def generate_intermediate_points(train_dataset):

    new_df = pd.DataFrame(columns = [f'x{i}' for i in range(1, train_dataset.shape[1] + 1)])

    for feat_idx in range(train_dataset.shape[1]): 

        new_feat_vals = []

        # Ordered feature values
        ordered_train_dataset = np.sort(train_dataset[:, feat_idx].flatten())

        for obs_idx in range(len(ordered_train_dataset) - 1):
            interm = uniform(ordered_train_dataset[obs_idx], ordered_train_dataset[obs_idx + 1])

            # Add newly generated obs
            new_feat_vals.append(interm)

        new_df[f'x{feat_idx + 1}'] = new_feat_vals

    # Add new observations to the existing train dataset
    return np.concatenate((train_dataset, new_df.to_numpy()), axis=0)