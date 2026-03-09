import numpy as np
# import matplotlib.pyplot as plt  #not used in this part
# import random                    #not used in this part

#discounting rate
gamma = 0.99

#reward outside of a terminal state
rewardNoTerminal = -1

#size of grid
gridSize = 4

#two terminal states
terminationStates = [[0,0], [gridSize-1, gridSize-1]]

#four actions: up, down, right, left
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]

#number of iterations we want to do
numIterations = 1000


def actionRewardFunction(initialPosition, action):
    #it returns the next state and reward
    #this function returns a reward of rewardNoTerminal each step unless you are in a terminal state
    #in that case, it returns a reward of zero

    #first check if we are in a termination state
    #in that case, reward is zero and the position remains the same
    if initialPosition in terminationStates:
        return initialPosition, 0

    #if we are not in a termination state, we return the variable rewardNoTerminal
    reward = rewardNoTerminal

    #calculate next position
    finalPosition = np.array(initialPosition) + np.array(action)

    #now check if the next position brings you out of the grid
    #if so, the finalposition should be the same as the initial position
    if -1 in finalPosition or gridSize in finalPosition:
        finalPosition = initialPosition

    return finalPosition, reward


#initialize value map to all zeros
valueMap = np.zeros((gridSize, gridSize))

#define the state space
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]

# deltas = []  #not used in this part
for it in range(numIterations):

    #make a copy of the value function to manipulate during the algorithm
    copyValueMap = np.copy(valueMap)

    # deltaState = []  #not used in this part
    for state in states:
        #Compute the Bellman iterate
        #V(s) = (1/4) * sum_a [ r(s,a) + gamma * V(s_next) ]
        #since pi(a|s) = 1/4 for all a (uniform random policy),
        #this is equivalent to taking the mean over all actions
        actionValues = []
        for action in actions:
            nextPosition, reward = actionRewardFunction(state, action)
            actionValues.append(reward + gamma * copyValueMap[nextPosition[0],
                                                      nextPosition[1]])
        valueMap[state[0], state[1]] = np.mean(actionValues)
        
    #for selected iterations, print the value function
    if it in [0, 1, 2, 9, 99, 199, 299, 399, 499, numIterations-1]:
        print("Iteration {}".format(it+1))
        print(valueMap)
        print("")