import sys
import os
import copy

# Made by Ao Wang, 15300240004 on May 14th, 2018

values = {-2:20, -1:0, 0:0, 1:0, 2:100} #the value of the state
actions = {-2:None, -1:None, 0:None, 1:None, 2:None}    #the optimal action of the state
gamma = 1.0

possible_state = [-1, 0, 1] #the states that values may change
iteration = 0
epsilon = 1.e-6
difference = 0.
reward = -5.    #the normal reward for each step
possible_action = [-1, +1]
action_realaction_possibility = {   #define the possibilities
    -1:{-1:0.8, +1:0.2},
    +1:{-1:0.7, +1:0.3}
}

while(iteration == 0 or difference > epsilon):
    
    '''if iteration == 2:
        print (actions)
        print (values)'''

    difference = 0.
    old_values = copy.deepcopy(values)
    for state in possible_state:
        value_action = []
        for action in possible_action:
            value = 0.
            for key in action_realaction_possibility:
                value += action_realaction_possibility[action][key] * old_values[state + key]
            value_action.append( (value, action) )
        v, a = max(value_action)
        values[state] = reward + gamma * v
        actions[state] = a
        difference = max(difference, abs(values[state] - old_values[state]))

    iteration+=1

print(actions)