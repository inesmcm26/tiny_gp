from random import random, randint, choice

from configs_bounded import *
from ops import FUNCTIONS, MAPPING, add, sub, mul, div

class GPTree:
    def __init__(self, node_value = None, left = None, right = None, terminals = None):
        self.node_value  = node_value
        self.left  = left
        self.right = right
        self.terminals = terminals
        self.tree_lambda = None
        
    def node_label(self):
        """
        Returns a string with the node label. Can be any terminal (variables) or non-terminal (functions).
        """
        if (self.node_value in FUNCTIONS):
            # Node is an op
            return self.node_value.__name__
        else: 
            # Node is a terminal
            return str(self.node_value)
        
        
    def node_depth(self, node_idx, current_idx = 1, current_depth = 0):
        if node_idx == current_idx:

            return current_idx, current_depth
        
        else:

            static_current_depth = current_depth

            if self.left:
                current_idx, current_depth = self.left.node_depth(node_idx, current_idx + 1, static_current_depth + 1)

            if self.right and current_idx != node_idx:
                current_idx, current_depth = self.right.node_depth(node_idx, current_idx + 1, static_current_depth + 1)
                
            return current_idx, current_depth
        
    def get_nodes_idx_above_depth(self, max_depth, current_node_idx = 1, current_depth = 0, nodes_list = []):
        
        if current_depth < max_depth:
            nodes_list.append(current_node_idx)
        
        if self.left:
            # print('GOING LEFT')
            current_node_idx, nodes_list =  self.left.get_nodes_idx_above_depth(max_depth, current_node_idx + 1, current_depth + 1, nodes_list)
        if self.right:
            # print('GOING RIGHT', self.node_label())
            current_node_idx, nodes_list =  self.right.get_nodes_idx_above_depth(max_depth, current_node_idx + 1, current_depth + 1, nodes_list)

        return current_node_idx, nodes_list

        
    def depth(self):

        if self.node_value in self.terminals:
            return 0
        
        return max(self.left.depth() + 1, self.right.depth() + 1)
        
    def print_tree(self, prefix = ""):
        """
        Prints the tree expression
        """
        print("%s%s" % (prefix, self.node_label()))   

        if self.left:
            self.left.print_tree(prefix + "   ")
        if self.right:
            self.right.print_tree(prefix + "   ")

    def save_tree_expression(self, prefix = ''):
        """
        Same as print_tree but returns the string to be printed

        Ignore
        """
        expr = prefix + self.node_label()

        if self.left:
            expr = expr + '\n' + self.left.save_tree_expression(prefix + '    ')
        if self.right:
            expr = expr + '\n' + self.right.save_tree_expression(prefix + '    ')

        return expr
    
    def create_expression(self):
        """
        Translated the tree to the corresponding algebraic expression
        """
        if self.node_value in FUNCTIONS:
            l = '(' + self.left.create_expression() + ')'
            r = '(' + self.right.create_expression() + ')'

            return l + ' ' + MAPPING[self.node_label()] + ' ' + r
        else:
            return self.node_value
        
    def tree2_string(self):
        if self.node_value in FUNCTIONS:
            l = '(' + self.left.tree2_string() + ')'
            r = '(' + self.right.tree2_string() + ')'

            return self.node_label() + f'({l}, {r})'

        else:
            return self.node_value

        
    def create_lambda_function(self):
        """
        Save lambda function as an attribute of the tree
        """

        string_expr = self.tree2_string()

        self.tree_lambda = eval(f'lambda {", ".join(self.terminals)}: {string_expr}')

        self.tree_lambda.expr = string_expr
        # self.expression = self.create_expression()

    def compute_tree(self, obs): 
        """
        Calculates the output of a given input

        Args:
            obs: np.array with observation values
        """

        # Lambda function version
        return self.tree_lambda(*obs)
    
        # # Node is a function
        # if self.node_value in FUNCTIONS: 
        #     return self.node_value(self.left.compute_tree(obs), self.right.compute_tree(obs))
        
        # # Node is a terminal variable
        # elif self.node_label().startswith('x'):

        #     # Get the variable index
        #     variable_idx = int(self.node_label()[1:])

        #     # Get the value of that variable
        #     return obs[variable_idx - 1] # Features are numbered from 1 to P

        # # Node is a terminal constant
        # else:
        #     return self.node_value
            
    def random_tree(self, grow, max_depth, depth = 0):
        """
        Create random tree using either grow or full method.
        This tree will be rooted on the current tree root node.
        """
        # Get a random function
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.node_value = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]

        # Get a random terminal
        elif depth >= max_depth:   
            self.node_value = self.terminals[randint(0, len(self.terminals)-1)]
        
        # Intermediate depth, grow
        else:
            rand_idx = randint(0, (len(self.terminals) + len(FUNCTIONS) - 1))

            if rand_idx < len(self.terminals):
                self.node_value = self.terminals[rand_idx]
            else:
                self.node_value = FUNCTIONS[rand_idx - len(self.terminals)]
            # if random () > 0.5: 
            #     self.node_value = self.terminals[randint(0, len(self.terminals)-1)]
            # else:
            #     self.node_value = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        
        # Generate sub trees
        if self.node_value in FUNCTIONS:
            self.left = GPTree(terminals = self.terminals)          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree(terminals = self.terminals)
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        """
        Standard one-point mutation
        """

        random_node_idx = randint(1, self.size())

        node_depth = self.node_depth(random_node_idx)[1]

        max_depth = MAX_DEPTH - node_depth

        new_subtree = GPTree(terminals = self.terminals)
        new_subtree.random_tree(grow = True, max_depth = max_depth)  

        self.scan_tree([random_node_idx], new_subtree)

        # self.create_safe_expression()
        self.create_lambda_function()

    def size(self):
        """
        Number of nodes
        """
        if self.node_value in self.terminals:
            return 1
        
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):
        """
        Builds a copy of the current tree
        """
        t = GPTree(terminals = self.terminals)
        t.node_value = self.node_value
        if self.left: 
            t.left  = self.left.build_subtree()
        if self.right: 
            t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"

        if count[0] == 1: 
            count[0] -= 1

            # If second tree, return the subtree rooted here
            if not second:
                new_tree = self.build_subtree()
                return new_tree
            
            # If first tree, replace with the second tree subtree
            else:
                self.node_value  = second.node_value
                self.left  = second.left
                self.right = second.right


        else:
            count[0] -= 1
            # Scan the rest of the tree to get to the desired node
            ret = None              
            if self.left:
                ret = self.left.scan_tree(count, second) 

            if self.right and count[0] > 0:
                ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): 
        """
        Crossover of 2 trees at random nodes
        """

        random_node_idx_second = randint(1, other.size())

        random_node_depth = other.node_depth(random_node_idx_second)[1]

        second_subtree = other.scan_tree([random_node_idx_second], None)

        second_subtree_depth = second_subtree.depth()

        nodes_above_depth = self.get_nodes_idx_above_depth(MAX_DEPTH - (second_subtree_depth + 1) + 2, nodes_list=[])

        search_nodes_idx = []

        for node_idx in nodes_above_depth[1]:
            subtree = self.scan_tree([node_idx], None)

            if MAX_DEPTH - (random_node_depth - 1) - 1 >= subtree.depth():
                search_nodes_idx.append(node_idx)

        
        # Scan first tree to get the subtree
        random_node_idx_first = choice(search_nodes_idx)
        first_subtree = self.scan_tree([random_node_idx_first], None)

        self.scan_tree([random_node_idx_first], second_subtree)
        other.scan_tree([random_node_idx_second], first_subtree)

        # self.create_safe_expression()
        # other.create_safe_expression()

        self.create_lambda_function()
        other.create_lambda_function()