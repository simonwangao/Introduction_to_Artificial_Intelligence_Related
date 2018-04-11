# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:42:24 2017

@author: Ting
@modified: Ao Wang, 15300240004
"""

inf = 999999999

class Node:
    def __init__(self, rule = 0, successor = [], isLeaf = False, value = None):
        if rule == 1:
            self.rule = 'max'
        else:
            self.rule = 'min'
        self.successor = successor
        self.isLeaf = isLeaf
        self.value = value
        self.visited = False


def value(node, alpha, beta):
    if node.rule == 'max':
        return maxValue(node, alpha, beta)
    if node.rule == 'min':
        return minValue(node, alpha, beta)


def maxValue(node, alpha, beta):
    node.visited = True
    if node.isLeaf:
        return node.value
    
    v = -inf
    child_list = node.successor
    for child in child_list:
        v = max(v, minValue(child, alpha, beta))
        if v >= beta:
            return v
        alpha = max(v, alpha)
    return v


def minValue(node, alpha, beta):
    node.visited = True
    if node.isLeaf:
        return node.value
    
    v = inf
    child_list = node.successor
    for child in child_list:
        v = min(v, maxValue(child, alpha, beta))
        if v <= alpha:
            return v
        beta = min(v, beta)
    return v

def unvisited(node):
    unvisit = []
    if node.successor:
        for successor in node.successor:
            unvisit += unvisited(successor)
    else:
        if not node.visited:
            unvisit.append(node.value)
    return unvisit


def constructTree(n, tree, rule):
    '''
    construct a tree using given information, and return the root node
    :param n:  the height of tree
    :param tree: the input tree described with list nested structure
    :param rule: root node's type, 1 for max, 0 for min
    :return: root node
    '''
    node = Node(rule=rule)
    successors = []
    if n == 1:
        for t in tree:
            successors.append(Node(rule=1-rule, isLeaf=True, value=t))
    else:
        for t in tree:
            successors.append(constructTree(n-1, t, 1-rule))
    node.successor = successors
    return node


while True:
    try:
        rule, n = map(int, input().strip().split())
        tree = eval(input().strip())
        root_node = constructTree(n-1, tree, rule)

        print(value(root_node, float("-inf"), float("inf"))) ## print out MINI-MAX value
        print(' '.join([str(node) for node in unvisited(root_node)]))  ## print out unvisited nodes
    except EOFError:
        break
