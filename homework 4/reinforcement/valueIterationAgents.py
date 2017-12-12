# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).



import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.
        """
        '''
        #Some useful mdp methods you will use:
        States 
        Action 
        Ttransition&prob 
        Reward 
        Y 
        Horizon - isTerminal
        Find optimal policy
        print(mdp.getStates())
        # ['TERMINAL_STATE', (0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)]
        
        hello = mdp.getStates()[10]
        
        print(hello)
        
        print(mdp.getPossibleActions(hello))
        #('north', 'west', 'south', 'east')
        
        mdp.getTransitionStatesAndProbs(hello, 'south')
        #mdp.getTransitionStatesAndProbs(hello, 'south') = [((1, 2), 0.8), ((0, 2), 0.1), ((2, 2), 0.1)]
        
        print(mdp.getReward(hello, 'south', (3,0)))
        #gives you the probability
    
        print(mdp.isTerminal(hello))
        
        '''
        
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0


        # Write value iteration code here
        # "*** YOUR CODE HERE ***"
        # state = mdp.getStates()
        # states = state[1:]
        # print values
        #value = []
        x = 0
        while( x < iterations):
            random = self.values.copy()
            # actions = mdp.getPossibleActions()
            states = mdp.getStates()
            for i in states:
                # print i
                # if(mdp.isTerminal(i)==True): its already initialized with zero
                if(mdp.isTerminal(i)==False):
                    summ = 0
                    maxx_qval = -100000
                    directions = mdp.getPossibleActions(i)
                    print directions
                    for j in directions:
                        summ = 0
                        trans = mdp.getTransitionStatesAndProbs(i,j)
                        # Transition probability
                        for k in trans:
                            # print k
                            # print k[0] #k[1] - probability 
                            myreward = mdp.getReward(i,j,k[0])
                            #reward
                            # print myreward
                            summ = summ+ ( ((discount*self.values[k[0]]) + myreward) * k[1])
                        qvalue = summ
                        if(qvalue>maxx_qval):
                            maxx_qval = qvalue
                        random[i] = maxx_qval
            self.values = random
            x = x+1

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        #"*** YOUR CODE HERE ***"

        trans = self.mdp.getTransitionStatesAndProbs(state,action)
        qval = 0
        count = 0
        summ = 0
        #considering all possibilities
        for i in trans:
            myreward = self.mdp.getReward(state,action,i[0])
            #summ
            count = count+1
            summ = summ + ( ((self.discount*self.values[i[0]]) + myreward) * i[1])
        return summ
        # util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # "*** YOUR CODE HERE ***"

        val = -10000
        count = 0
        pol = None
        directions = self.mdp.getPossibleActions(state)
        if self.mdp.isTerminal(state) == True:
            return None
        for i in directions:

            count = 0
            qval = self.getQValue(state,i)
            if qval > val:
                pol = i
                val = qval
                count = count+1
        return pol
        # return 0
        # util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
