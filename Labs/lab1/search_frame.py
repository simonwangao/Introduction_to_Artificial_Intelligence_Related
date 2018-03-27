import heapq

class PriorityQueue(object):
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        self.count += 1
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        self.count -= 1
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)

class Node:
    """define node"""
    def __init__(self, state, parent, path_cost, action):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.action = action

class Problem:
    """searching problem"""
    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.actions = actions
        # you can add code here or not

    def search_actions(self, state):
        raise Exception('get all the possible actions for the state')
        
    def solution(self, node):
        raise Exception('get the path from start node to current node')

    def transition(self, state, action):
        raise Exception('the finished state after the state takes the action')

    def goal_test(self, state):
        raise Exception('to judge whether the node is the goal')
    
    def step_cost(self, state1, action, state2):
        raise Exception('get the cost from state1 to state2 after taking the action')

    def child_node(self, node_begin, action):
        raise Exception('get the node after node_begin takes the action')

def UCS(problem):
    node_test = Node(problem.initial_state, '', 0, '')
    frontier = PriorityQueue()
    frontier.push(node_test, node_test.path_cost)
    explored = []
    raise Exception('start the loop')

def main():
        actions = []
        while True:
            a = input().strip()
            if a != 'END':
                a = a.split()
                actions += [a]
            else:
                break
        graph_problem = Problem('Start', actions)
        answer = UCS(graph_problem) 
        s = "->"
        if answer == 'Unreachable':
            print(answer)
        else:
            path = s.join(answer)
            print(path)

if __name__=='__main__':
    main()