import heapq

class PriorityQueue(object):
    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        self.count += 1
        entry = (priority, self.count, item)
        self.heap.append(entry)

    def pop(self):
        temp=self.heap[0]
        temp_index=0      
        for index, it in enumerate(self.heap):
            if temp[0] > it[0]:   #record the smallest one
                temp=it[:]
                temp_index=index
        del self.heap[temp_index]
        self.count-=1
        return temp[2]
        
    def isEmpty(self):
        return len(self.heap) == 0
    
    def isIn(self, state): #
        for thing in self.heap:
            if state == thing[2].state:
                return True
        return False
    
    def update(self, item, priority):
        for index, (p, c, i) in enumerate(self.heap):
            if i.state == item.state:
                if p <= priority:   #change
                    return
                del self.heap[index]
                self.heap.append((priority, c, item))
                return 
        self.push(item, priority)

class Node:
    """define node"""
    def __init__(self, state, parent, path_cost, actions):
        self.state = state
        self.parent = parent
        self.path_cost = path_cost
        self.actions = actions

class problem:
    """searching problem"""
    def __init__(self, initial_state, actions):
        self.initial_state = initial_state
        self.actions = actions
        # 可以在这里随意添加代码或者不加
            
    def search_actions(self, state):
        state_possible_actions=[]
        for pair in self.actions:
            if pair[0] == state:
                state_possible_actions.append(pair)
        return state_possible_actions
        
    def solution(self, node):
        path=[]
        path.append(node.state)
        while node.state != 'Start':
            path.append(node.state)
            node=node.parent
        path.append(self.initial_state)
        path=path[-1:0:-1]
        return path

    def transition(self, state, action):
        if action[0] == state:
            return action[1]
        else:
            return "Wrong"

    def goal_test(self, state):
        if state == "Goal":
            return True
        else:
            return False

def UCS(problem):
    node_test = Node(problem.initial_state, '', 0, '')
    frontier = PriorityQueue()
    frontier.push(node_test, node_test.path_cost)
    explored = []
    while(True):
        if frontier.isEmpty()==True:
            return 'Unreachable'
        current_node=frontier.pop()
        if problem.goal_test(current_node.state)==True:
            return problem.solution(current_node)
        explored.append(current_node.state)

        temp_actions = problem.search_actions(current_node.state)
        for action in temp_actions:
            child_state=problem.transition(current_node.state,action)
            if child_state == "Wrong":
                continue
            if child_state not in explored and frontier.isIn(child_state) == False: #
                cost=current_node.path_cost+int(action[2])
                s_actions=problem.search_actions(child_state)
                node=Node(child_state, current_node, cost, s_actions)
                frontier.push(node,node.path_cost)
            elif frontier.isIn(child_state) == True:
                cost=current_node.path_cost+int(action[2])
                s_actions=problem.search_actions(child_state)
                node=Node(child_state, current_node, cost, s_actions)
                frontier.update(node, current_node.path_cost+int(action[2]))


def main():
        actions = []
        while True:
            a = input()
            a=a.strip()
            if a != 'END':
                a = a.split()
                actions += [a]
            else:
                break
        graph_problem = problem('Start', actions)
        answer = UCS(graph_problem) 
        s = "->"
        if answer == 'Unreachable':
            print(answer)
        else:
            path = s.join(answer)
            print(path)

if __name__=='__main__':
    main()
