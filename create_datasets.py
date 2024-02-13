import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

FILE = '/home/ines/Downloads/BostonHousing.csv'

random_states = np.random.choice(100, 30)

df = pd.read_csv(FILE)

for i in range(1, 31):
    train, test = train_test_split(df, test_size=0.3, random_state = random_states[i - 1], shuffle = True)
    train.reset_index(drop=True, inplace=True)
    test.reset_index(drop=True, inplace=True)
    train.to_csv(f'/home/ines/Documents/tese/tiny_gp/data/BHousing/train_{i}.csv')
    test.to_csv(f'/home/ines/Documents/tese/tiny_gp/data/BHousing/test_{i}.csv')
