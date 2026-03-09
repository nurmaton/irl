import numpy as np

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
action_names = ['U', 'D', 'R', 'L']  # Corresponding letters for printing the policy

#define the state space
states = [[i, j] for i in range(gridSize) for j in range(gridSize)]


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


def policy_evaluation(policy, tol=1e-6):
    #Evaluate a given policy until convergence (max change < tol)
    valueMap = np.zeros((gridSize, gridSize))
    while True:
        copyValueMap = np.copy(valueMap)
        delta = 0
        for state in states:
            action = actions[policy[state[0], state[1]]]
            nextPosition, reward = actionRewardFunction(state, action)
            new_value = reward + gamma * copyValueMap[nextPosition[0], nextPosition[1]]
            delta = max(delta, abs(new_value - valueMap[state[0], state[1]]))
            valueMap[state[0], state[1]] = new_value
        if delta < tol:
            break
    return valueMap


def policy_improvement(valueMap):
    #Improve the policy greedily with respect to the current value function
    newPolicy = np.zeros((gridSize, gridSize), dtype=int)
    for state in states:
        actionValues = []
        for action in actions:
            nextPosition, reward = actionRewardFunction(state, action)
            actionValues.append(reward + gamma * valueMap[nextPosition[0], nextPosition[1]])
        newPolicy[state[0], state[1]] = np.argmax(actionValues)
    return newPolicy


#Start with the policy that always goes down (action index 1 = 'D')
policy = np.ones((gridSize, gridSize), dtype=int)

print("Initial policy (always Down):")
print(np.vectorize(lambda x: action_names[x])(policy))
print()

iteration = 0
while True:
    #Step 1: policy evaluation
    valueMap = policy_evaluation(policy)

    #Step 2: policy improvement
    newPolicy = policy_improvement(valueMap)

    iteration += 1
    print("Policy Improvement Iteration {}:".format(iteration))
    print("Value Map:")
    print(valueMap)
    print("New Policy:")
    print(np.vectorize(lambda x: action_names[x])(newPolicy))
    print()

    #Step 3: check for convergence
    if np.array_equal(newPolicy, policy):
        print("Policy has converged!")
        break
    policy = newPolicy