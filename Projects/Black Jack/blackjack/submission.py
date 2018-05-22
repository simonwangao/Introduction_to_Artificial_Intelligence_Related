import util, math, random
from collections import defaultdict
from util import ValueIteration

############################################################
# Problem 2a

# If you decide 2a is true, prove it in blackjack.pdf and put "return None" for
# the code blocks below.  If you decide that 2a is false, construct a counterexample.
class CounterexampleMDP(util.MDP):
    def startState(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return 0
        # END_YOUR_CODE

    # Return set of actions possible from |state|.
    def actions(self, state):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if state == 0:
            return [-1]
        if state == 1:
            return [-2]
        if state == -1: #exit
            return [0]
        # END_YOUR_CODE

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if state == 0:
            t0_original = 0 # original one has no chance to get to state 0 with positive reward
            t1_original = 0 # original one has no chance to get to state 1 with high reward
            t_1_original = 1 # original one will definitely go to state -1 with low reward
            t1 = 0.5 * t1_original + 0.5 * 1./3.    # it now has chance to go to a state with high reward
            t_1 = 0.5 * t_1_original + 0.5 * 1./3.
            t0 = 0.5 * t0_original + 0.5 * 1./3.
            return [(-1, t_1, -20), (1, t1, 20), (0, t0, 10)]
        if state == 1:
            t1_original = 0 # the original one will have no possibility to go to state 1 with high reward
            t_1_original = 1 # the original one will definitely go to state -1 with low reward
            t0_original = 0 # the original one will have no possibility to go to state 0 with positive reward
            t_1 = 0.5 * t_1_original + 0.5 * 1./3.
            t0 = 0.5 * t0_original + 0.5 * 1./3.
            t1 = 0.5 * t1_original + 0.5 * 1./3. # it now has chance to go to a state with high reward
            return [(-1, t_1, -20), (0, t0, 10), (1, t1, 20)]
        if state == -1:
            return []
        # END_YOUR_CODE

    def discount(self):
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        return 1.0
        # END_YOUR_CODE

############################################################
# Problem 3a

class BlackjackMDP(util.MDP):
    def __init__(self, cardValues, multiplicity, threshold, peekCost):
        """
        cardValues: array of card values for each card type
        multiplicity: number of each card type
        threshold: maximum total before going bust
        peekCost: how much it costs to peek at the next card
        """
        self.cardValues = cardValues
        self.multiplicity = multiplicity
        self.threshold = threshold
        self.peekCost = peekCost

    # Return the start state.
    # Look at this function to learn about the state representation.
    # The first element of the tuple is the sum of the cards in the player's
    # hand.
    # The second element is the index (not the value) of the next card, if the player peeked in the
    # last action.  If they didn't peek, this will be None.
    # The final element is the current deck.
    def startState(self):
        return (0, None, (self.multiplicity,) * len(self.cardValues))  # total, next card (if any), multiplicity for each card

    # Return set of actions possible from |state|.
    # You do not need to modify this function.
    # All logic for dealing with end states should be done in succAndProbReward
    def actions(self, state):
        return ['Take', 'Peek', 'Quit']

    # Return a list of (newState, prob, reward) tuples corresponding to edges
    # coming out of |state|.  Indicate a terminal state (after quitting or
    # busting) by setting the deck to None. 
    # When the probability is 0 for a particular transition, don't include that 
    # in the list returned by succAndProbReward.
    def succAndProbReward(self, state, action):
        # BEGIN_YOUR_CODE (our solution is 53 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        current_value, top_card, old_deck = state
        if old_deck is None or sum(old_deck) == 0:  #exit by empty actions
            return []

        if action == 'Quit':
            new_state = (current_value, None, None)
            return [(new_state, 1.0, current_value)]    #current value as the reward
        
        if action == 'Peek':
            if top_card is not None:    #can not continually peek
                return []
            num = sum(old_deck)
            res = []
            for i in range(len(self.cardValues)):
                if old_deck[i] > 0: # not be zero
                    new_state = (current_value, i, old_deck)
                    res.append( (new_state, float(old_deck[i])/num, -self.peekCost) )
            return res
        
        if action == 'Take':
            if top_card is not None:
                new_value = current_value + self.cardValues[top_card]   # last turn is peeked, so the result is settled
                if new_value <= self.threshold:
                    deck = list(old_deck[:])
                    deck[top_card] -= 1
                    deck = tuple(deck)
                    if sum(deck) != 0:
                        new_state = (new_value, None, deck) #still cards left
                        return [(new_state, 1.0, 0)]
                    else:
                        new_state = (new_value, None, None) #no card left
                        return [(new_state, 1.0, new_state[0])]
                else:
                    new_state = (new_value, None, None) #strictly greater than the threshold
                    return [(new_state, 1.0, 0)]
            
            res_lis = []
            num = sum(old_deck)
            for i in range(len(self.cardValues)):
                if old_deck[i] != 0: #still have cards
                    new_value = current_value + self.cardValues[i]
                    if new_value <= self.threshold:
                        new_deck = list(old_deck[:])
                        new_deck[i] -= 1
                        new_deck = tuple(new_deck)
                        if sum(new_deck) != 0:
                            new_state = (new_value, None, new_deck)
                            res_lis.append((new_state, float(old_deck[i])/num, 0))  # normal
                        else:
                            new_state = (new_value, None, None)
                            res_lis.append((new_state, float(old_deck[i])/num, new_state[0]))
                    else:
                        new_state = (new_value, None, None)
                        res_lis.append((new_state, float(old_deck[i])/num, 0))  #
            return res_lis

        # END_YOUR_CODE

    def discount(self):
        return 1

############################################################
# Problem 3b

def peekingMDP():
    """
    Return an instance of BlackjackMDP where peeking is the optimal action at
    least 10% of the time.
    """
    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    mdp = BlackjackMDP(cardValues=[2,3,4,15], multiplicity=3, threshold=20, peekCost=1)
    return mdp
    # END_YOUR_CODE

############################################################
# Problem 4a: Q learning

# Performs Q-learning.  Read util.RLAlgorithm for more information.
# actions: a function that takes a state and returns a list of actions.
# discount: a number between 0 and 1, which determines the discount factor
# featureExtractor: a function that takes a state and action and returns a list of (feature name, feature value) pairs.
# explorationProb: the epsilon value indicating how frequently the policy
# returns a random action
class QLearningAlgorithm(util.RLAlgorithm):
    def __init__(self, actions, discount, featureExtractor, explorationProb=0.2):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.explorationProb = explorationProb
        self.weights = defaultdict(float)
        self.numIters = 0

    # Return the Q function associated with the weights and features
    def getQ(self, state, action):
        score = 0
        for f, v in self.featureExtractor(state, action):
            score += self.weights[f] * v
        return score

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        self.numIters += 1
        if random.random() < self.explorationProb:
            return random.choice(self.actions(state))
        else:
            return max((self.getQ(state, action), action) for action in self.actions(state))[1]

    # Call this function to get the step size to update the weights.
    def getStepSize(self):
        return 1.0 / math.sqrt(self.numIters)

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, state, action, reward, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        #raise Exception("Not implemented yet")
        if newState is not None:    # not the terminal state
            max_ = max(self.getQ(newState, new_action) for new_action in self.actions(newState))    # the max Q(s', a')
            sample = reward + self.discount * max_  # reward + gamma * max( Q(s', a') )
        else:
            sample = reward # the terminal state, only reward
        
        alpha = self.getStepSize()  # the learning rate, getting smaller every time
        old_Q = self.getQ(state, action)    # the old Q value

        # NOTE: the feature is actually the value of Q(s, a)
        for f, v in self.featureExtractor(state, action):
            self.weights[f] = (1 - alpha)*self.weights[f] + alpha*sample    # standard Q-learning
        # END_YOUR_CODE

# Return a singleton list containing indicator feature for the (state, action)
# pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]

############################################################
# Problem 4b: convergence of Q-learning
def test1():
    '''
    The reason of high error rate is that
    the learning rate is fixedly related to the number of iterations,
    so the learning rate may turn to 0 before the Q-learning is converged
    '''
    # Small test case
    smallMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)
    smallMDP.computeStates()

    # Large test case
    largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
    largeMDP.computeStates()

    mdp = largeMDP  #change
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                                        identityFeatureExtractor,
                                        0.2)
    
    # start value iteration
    vl = util.ValueIteration()
    vl.solve(mdp)
    vl_policy_dic = vl.pi
    #print (vl_policy_dic)

    # start Q-learning
    res = util.simulate(mdp, rl, numTrials=30000)
    rl_policy_dic = {}
    rl.explorationProb = 0.0
    for key in vl_policy_dic:
        rl_policy_dic[key] = rl.getAction(key)

    flag = 'Yes!'
    wrong = 0
    for key in rl_policy_dic:
        if key[2] is not None:
            if rl_policy_dic[key] != vl_policy_dic[key]:
                flag = 'No!'
                wrong += 1
    print ('With original identityFeatureExtractor:')
    print (flag, wrong)
    print (len(rl_policy_dic))
    print ('\n')


############################################################
# Problem 4c: features for Q-learning.

# You should return a list of (feature key, feature value) pairs (see
# identityFeatureExtractor()).
# Implement the following features:
# - indicator on the total and the action (1 feature).
# - indicator on the presence/absence of each card and the action (1 feature).
#       Example: if the deck is (3, 4, 0 , 2), then your indicator on the presence of each card is (1,1,0,1)
#       Only add this feature if the deck != None
# - indicator on the number of cards for each card type and the action (len(counts) features).  Only add these features if the deck != None
def blackjackFeatureExtractor(state, action):
    total, nextCard, counts = state
    # BEGIN_YOUR_CODE (our solution is 9 lines of code, but don't worry if you deviate from this)
    #raise Exception("Not implemented yet")
    res = []
    # 1st indicator:
    res.append( ((total, action), 1.0) )
    # 2nd indicator:
    if counts is not None:
        tup = tuple([1 if num else 0 for num in counts])
        res.append( ((tup, action), 1.0) )
    # 3rd indicator:
    if counts is not None:
        for i in range(len(counts)):
            res.append( ((i, counts[i], action), 1.0) )
    return res
    # END_YOUR_CODE

def test2():
    # test on largeMDP
    largeMDP = BlackjackMDP(cardValues=[1, 3, 5, 8, 10], multiplicity=3, threshold=40, peekCost=1)
    largeMDP.computeStates()

    mdp = largeMDP
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                                        blackjackFeatureExtractor,
                                        0.2)

    # start value iteration
    vl = util.ValueIteration()
    vl.solve(mdp)
    vl_policy_dic = vl.pi

    # start Q-learning
    res = util.simulate(mdp, rl, numTrials=30000)
    rl_policy_dic = {}
    rl.explorationProb = 0.0
    for key in vl_policy_dic:
        rl_policy_dic[key] = rl.getAction(key)

    flag = 'Yes!'
    wrong = 0
    for key in rl_policy_dic:
        if key[2] is not None:
            if rl_policy_dic[key] != vl_policy_dic[key]:
                flag = 'No!'
                wrong += 1
    print ('With blackjackFeatureExtractor:')
    print (flag, wrong)
    print (len(rl_policy_dic))

############################################################
# Problem 4d: What happens when the MDP changes underneath you?!

def test3():
    # Original mdp
    originalMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=10, peekCost=1)

    # New threshold
    newThresholdMDP = BlackjackMDP(cardValues=[1, 5], multiplicity=2, threshold=15, peekCost=1)

    # start value iteration
    vl = util.ValueIteration()
    vl.solve(originalMDP)
    vl_policy_dic = vl.pi

    fixed_RL = util.FixedRLAlgorithm(vl_policy_dic)
    res = util.simulate(newThresholdMDP, fixed_RL, numTrials=300)
    print (res)

    # start Q-learning
    mdp = newThresholdMDP
    rl = QLearningAlgorithm(mdp.actions, mdp.discount(),
                                        blackjackFeatureExtractor,
                                        0.2)
    res = util.simulate(mdp, rl, numTrials=300)
    print (res)



if __name__ == '__main__':
    test1()
    test2()
    test3()