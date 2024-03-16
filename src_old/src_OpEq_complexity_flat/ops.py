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

FUNCTIONS = [add, sub, mul, div]

MAPPING = {
    'add': '+',
    'sub': '-',
    'mul': '*',
    'div': '/'
}