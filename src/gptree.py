from random import random, randint, choice

from configs import *
from ops import FUNCTIONS, MAPPING

class GPTree:
    def __init__(self, node_value = None, left = None, right = None, terminals = None):
        self.node_value  = node_value
        self.left  = left
        self.right = right
        self.terminals = terminals
        
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
        
    # def node_depth(self, node_idx, current_depth = 0):
    #     if node_idx[0] == 1:
    #         node_idx[0] -= 1

    #         # print('REACHED NODE')
    #         # print(self.node_label())
    #         # print('DEPTH', current_depth)
    #         return current_depth
        
    #     else:
        
    #         node_idx[0] -= 1
            
    #         ret = None              
    #         if self.left:
    #             # print('GOING LEFT')
    #             ret = self.left.node_depth(node_idx, current_depth + 1)
    #         if self.right and node_idx[0] > 0:
    #             # print('GOING RIGHT. CURRENT NODE IDX', node_idx[0])
    #             ret = self.right.node_depth(node_idx, current_depth + 1)
            
    #         return ret
        
    def node_depth(self, node_idx, current_idx = 1, current_depth = 0):
        if node_idx == current_idx:

            # print('REACHED NODE')
            # print(self.node_label())
            # print('DEPTH', current_depth)
            return current_idx, current_depth
        
        else:

            static_current_depth = current_depth

            if self.left:
                # print('GOING LEFT')
                # print('CURRENT NODE LABEL', self.node_label())
                current_idx, current_depth = self.left.node_depth(node_idx, current_idx + 1, static_current_depth + 1)

            if self.right and current_idx != node_idx:
                # print('GOING RIGHT. CURRENT NODE IDX', current_idx)
                # print('CURRENT NODE LABEL', self.node_label())
                current_idx, current_depth = self.right.node_depth(node_idx, current_idx + 1, static_current_depth + 1)
                
            return current_idx, current_depth
        
    def get_nodes_idx_above_depth(self, max_depth, current_node_idx = 1, current_depth = 0, nodes_list = []):
        
        # print('CURRENT NODE:', self.node_label())
        # print('CURRENT NODE IDX', current_node_idx)
        # print('CURRENT DEPTH', current_depth)
        if current_depth < max_depth:
            nodes_list.append(current_node_idx)
        #     print('MAX DEPTH EXCEEDED')
        #     return current_node_idx, nodes_list
        
        # else:
            # print('NEW NODES LIST', nodes_list)
        
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

    def compute_tree(self, obs): 
        """
        Calculates the output of a given input

        Args:
            obs: np.array with observation values
        """

        # Node is a function
        if self.node_value in FUNCTIONS: 
            return self.node_value(self.left.compute_tree(obs), self.right.compute_tree(obs))
        
        # Node is a terminal variable
        elif self.node_label().startswith('x'):

            # print('NODE STARTS WITH X')
            # print(self.node_label())

            # Get the variable index
            variable_idx = int(self.node_label()[1:])
            # print('VARIABLE IDX', variable_idx)
            # print(obs)
            # Get the value of that variable
            return obs[variable_idx - 1] # Features are numbered from 1 to P

        # Node is a terminal constant
        else:
            return self.node_value
            
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
            if random () > 0.5: 
                self.node_value = self.terminals[randint(0, len(self.terminals)-1)]
            else:
                self.node_value = FUNCTIONS[randint(0, len(FUNCTIONS)-1)]
        
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

        # print('ORIGINAL TREE')
        # self.print_tree()

        random_node_idx = randint(1, self.size())

        # print('RANDOM NODE IDX', random_node_idx)
        
        node_depth = self.node_depth(random_node_idx)[1]

        # print('NODE DEPTH', node_depth)

        max_depth = MAX_DEPTH - node_depth

        # print('MAX DEPTH', max_depth)

        new_subtree = GPTree(terminals = self.terminals)
        new_subtree.random_tree(grow = True, max_depth = max_depth)

        # print('NEW SUBTREE')
        # new_subtree.print_tree()        

        self.scan_tree([random_node_idx], new_subtree)

        # print('MUTATED TREE')
        # self.print_tree()

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
        # print('------------')
        
        # Scan second tree to get the subtree
        random_node_idx_second = randint(1, other.size())

        # print('SECOND TREE NODE IDX', random_node_idx_second)

        random_node_depth = other.node_depth(random_node_idx_second)[1]

        # print('SECOND TREE RANDOM NODE DEPTH', random_node_depth)

        # print('CHOOSEN NODE FROM SECOND TREE', random_node_idx_second)
        second_subtree = other.scan_tree([random_node_idx_second], None)

        # print('SECOND SUBTREE')
        # second_subtree.print_tree()

        second_subtree_depth = second_subtree.depth()

        # print('SECOND SUBTREE DEPTH', second_subtree_depth)

        nodes_above_depth = self.get_nodes_idx_above_depth(MAX_DEPTH - (second_subtree_depth + 1) + 2, nodes_list=[])

        # print('NODES FROM FIRST TREE ABOVE DEPTH', nodes_above_depth)

        search_nodes_idx = []

        for node_idx in nodes_above_depth[1]:
            # print('UPPER BOUND = Maxima depth da primera subtree', MAX_DEPTH - (random_node_depth - 1) - 1)
            # print('CURRENT NODE', node_idx)
            subtree = self.scan_tree([node_idx], None)
            # print('NODE DEPTH', subtree.depth())
            if MAX_DEPTH - (random_node_depth - 1) - 1 >= subtree.depth():
                search_nodes_idx.append(node_idx)
            # else:
            #     print('-------------------------------------------')
            #     print('-------------------------------------------')
            #     print('-------------------------------------------')
            #     print('IMPOSSIBLE NODE')
            #     print('NODE IDX', node_idx)
            #     print('NODE DEPTH', subtree.depth())
            #     print('SUBTREE')
            #     subtree.print_tree()
            #     print('-------------------------------------------')
            #     print('-------------------------------------------')
            #     print('-------------------------------------------')

        # print('SEARCH NODES', search_nodes_idx)

        # print('SECOND SUBTREE')
        # second_subtree.print_tree()
        
        # Scan first tree to get the subtree
        random_node_idx_first = choice(search_nodes_idx)
        # print('CHOOSEN RANDOM NODE', random_node_idx_first)
        # print('CHOOSEN NODE FROM FIRST TREE', random_node_idx_first)
        first_subtree = self.scan_tree([random_node_idx_first], None)

        # print('FIRST SUBTREE')
        # first_subtree.print_tree()

        # print('FIRST SUBTREE')
        # first_subtree.print_tree()

        self.scan_tree([random_node_idx_first], second_subtree)
        other.scan_tree([random_node_idx_second], first_subtree)

        # print('FIRST TREE AFTER')
        # self.print_tree()
        # print('SECOND TREE AFTER')
        # other.print_tree()