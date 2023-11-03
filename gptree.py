from random import random, randint

from configs import *
from ops import FUNCTIONS

class GPTree:
    def __init__(self, node_value = None, left = None, right = None):
        self.node_value  = node_value
        self.left  = left
        self.right = right
        
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
    
    def print_tree(self, prefix = ""):
        """
        Prints the tree expression
        """
        print("%s%s" % (prefix, self.node_label()))   

        if self.left:
            self.left.print_tree (prefix + "   ")
        if self.right:
            self.right.print_tree(prefix + "   ")

    def compute_tree(self, obs): 
        """
        Calculates the output of a given input

        Args:
            obs: np.array with observation values
        """


        # Node is a function
        if (self.node_value in FUNCTIONS): 
            return self.node_value(self.left.compute_tree(obs), self.right.compute_tree(obs))
        
        # Node is a terminal variable
        elif self.node_value.startswith('x'):
            # Get the variable index
            variable_idx = int(self.node_value[1:])
            # Get the value of that variable
            return obs[variable_idx]

        # Node is a terminal constant
        else:
            return self.node_value
            
    def random_tree(self, grow, max_depth, depth = 0):
        """
        Create random tree using either grow or full method
        """
        # Get a random function
        if depth < MIN_DEPTH or (depth < max_depth and not grow): 
            self.node_value = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]

        # Get a random terminal
        elif depth >= max_depth:   
            self.node_value = TERMINALS[randint(0, len(TERMINALS)-1)]
        
        # Intermediate depth, grow
        else:
            if random () > 0.5: 
                self.node_value = TERMINALS[randint(0, len(TERMINALS)-1)]
            else:
                self.node_value = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        
        # Generate sub trees
        if self.node_value in FUNCTIONS:
            self.left = GPTree()          
            self.left.random_tree(grow, max_depth, depth = depth + 1)            
            self.right = GPTree()
            self.right.random_tree(grow, max_depth, depth = depth + 1)

    def mutation(self):
        """
        Standard one-point mutation
        """

        random_node_idx = [randint(1, self.size())]

        new_subtree = GPTree()
        new_subtree.random_tree(grow = True, max_depth = 2) # TODO: change here the mutation hyperparameters

        # print('NEW SUBTREE')
        # new_subtree.print_tree()

        # print('NODE INDEX:', random_node_idx)
        
        # print('ORIGINAL TREE')
        # self.print_tree()

        self.scan_tree(random_node_idx, new_subtree)

        # print('MUTATED TREE')
        # self.print_tree()

    def size(self):
        """
        Number of nodes
        """
        if self.node_value in TERMINALS:
            return 1
        
        l = self.left.size()  if self.left  else 0
        r = self.right.size() if self.right else 0
        return 1 + l + r

    def build_subtree(self):
        """
        Builds a copy of the current tree
        """
        t = GPTree()
        t.node_value = self.node_value
        if self.left: 
            t.left  = self.left.build_subtree()
        if self.right: 
            t.right = self.right.build_subtree()
        return t
                        
    def scan_tree(self, count, second): # note: count is list, so it's passed "by reference"

        # print('SCANNING')
        # self.print_tree()
        # print('COUNT', count)

        if count[0] == 1: 
            count[0] -= 1
            # print('COUNT == 1')

            # If second tree, return the subtree rooted here
            if not second:
                # print('SECOND. build subtree')
                new_tree = self.build_subtree()

                # print('NEW TREE')
                # new_tree.print_tree()
                return new_tree
            
            # If first tree, replace with the second tree subtree
            else:
                self.node_value  = second.node_value
                self.left  = second.left
                self.right = second.right

                # print('NEW TREE')
                # self.print_tree()

        else:
            count[0] -= 1
            # print('SCANNING REST OF THE TREE')
            # Scan the rest of the tree to get to the desired node
            ret = None              
            if self.left:
                # print('SCAN LEFT')
                ret = self.left.scan_tree(count, second) 

            if self.right and count[0] > 0:
                # print('SCAN RIGHT')
                ret = self.right.scan_tree(count, second)  
            return ret

    def crossover(self, other): 
        """
        Crossover of 2 trees at random nodes
        """

        # print('CROSSOVER')
        # print('FIRST TREE')
        # self.print_tree()
        # print('SECOND TREE')
        # other.print_tree()
        
        # Scan second tree to get the subtree
        random_node_idx_second = randint(1, other.size())
        # print('CHOOSEN NODE FROM SECOND TREE', random_node_idx_second)
        second_subtree = other.scan_tree([random_node_idx_second], None) # 2nd random subtree

        # print('SECOND SUBTREE')
        # second_subtree.print_tree()
        
        # Scan first tree to get the subtree
        random_node_idx_first = randint(1, self.size())
        # print('CHOOSEN NODE FROM FIRST TREE', random_node_idx_first)
        first_subtree = self.scan_tree([random_node_idx_first], None) # 2nd random subtree

        # print('FIRST SUBTREE')
        # first_subtree.print_tree()

        self.scan_tree([random_node_idx_first], second_subtree)
        other.scan_tree([random_node_idx_second], first_subtree)

        # print('FIRST TREE AFTER')
        # self.print_tree()
        # print('SECOND TREE AFTER')
        # other.print_tree()