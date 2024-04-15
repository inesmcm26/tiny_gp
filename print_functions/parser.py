from gptree import GPTree

def tokenize(s : str) -> list[str]:
    tokens = []
    lexeme = ''
    for ch in s:
        if ch == ' ':
            if lexeme:
                tokens.append(lexeme)
                lexeme = ''
            continue
        elif not (ch.isalpha() or ch.isdigit()):
            if lexeme:
                tokens.append(lexeme)
                lexeme = ''
            tokens.append(ch)
        else:
            lexeme += ch
    return tokens


def parse_operator(op: str) -> int:
    from ops import add, sub, mul, div
    match op:
        case '+':
            return (add, 1)
        case '-':
            return (sub, 1)
        case '*':
            return (mul, 2)
        case '/':
            return (div, 2)
        case _:
            return (None, 0)

def parse(tokens, terminals, precedence= 0):
    lhs = parse_prefix(tokens, terminals)
    while True:
        if not tokens:
            break
        op, prec = parse_operator(tokens[0])
        if op is None or prec < precedence:
            break
        tokens.pop(0)
        lhs = GPTree(op, lhs, parse(tokens, prec), terminals=terminals)
    return lhs

def parse_prefix(tokens, terminals):
    match tokens[0]:
        case '(':
            tokens.pop(0)
            tree = parse(tokens, terminals=terminals, precedence=0)
            assert tokens.pop(0) == ')'
            return tree
        case _:
            assert tokens[0].isidentifier()
            return GPTree(tokens.pop(0), terminals=terminals)
