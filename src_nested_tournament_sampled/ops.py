import  numpy as np

def add(x, y):
    return x + y

def sub(x, y):
    return x - y

def mul(x, y):
    return x * y

def div(x, y):
    # Edge case
    if y == 0:
        return 1
    
    return x / y

def cos(x):
    return np.cos(x)

def sin(x):
    return np.sin(x)

def sqrt(x):
    return np.sqrt(x)

FUNCTIONS = [add, sub, mul, div]

MAPPING = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/',
}