import numpy as np

# Grid dimensions and variables
gridSize = 4
numStates = gridSize * gridSize
rewardNoTerminal = -1
gamma = 0.99

# Terminal states (0-indexed: 0 is top-left, 15 is bottom-right)
terminationStates = [0, numStates - 1]

# Four actions: Up, Down, Right, Left
actions = [[-1, 0], [1, 0], [0, 1], [0, -1]]
action_prob = 1.0 / len(actions) # Uniform random policy

# --- 1. Construct P_pi ---
P_pi = np.zeros((numStates, numStates))

for s in range(numStates):
    if s in terminationStates:
        P_pi[s, s] = 1.0
        continue
        
    row = s // gridSize
    col = s % gridSize
    
    for action in actions:
        next_row = row + action[0]
        next_col = col + action[1]
        
        if next_row < 0 or next_row >= gridSize or next_col < 0 or next_col >= gridSize:
            P_pi[s, s] += action_prob
        else:
            next_s = next_row * gridSize + next_col
            P_pi[s, next_s] += action_prob

# --- 2. Construct r_pi ---
R_pi = np.zeros((numStates, 1))

for s in range(numStates):
    if s in terminationStates:
        R_pi[s, 0] = 0.0
    else:
        # Expected reward: sum of (probability * reward) for all actions
        expected_reward = 0
        for action in actions:
            expected_reward += action_prob * rewardNoTerminal
        R_pi[s, 0] = expected_reward

# --- 3. Solve v_pi = (I - gamma * P_pi)^-1 * r_pi ---
I = np.eye(numStates)
A = I - gamma * P_pi
A_inv = np.linalg.inv(A)

V_pi_1D = np.dot(A_inv, R_pi)

# Reshape the 16x1 vector back into the 4x4 grid format
V_pi_grid = V_pi_1D.reshape((gridSize, gridSize))

# --- Print Results ---
np.set_printoptions(linewidth=200, formatter={'float': '{:0.8f}'.format})
print("Exact Analytical Value Function V_pi (4x4):")
print(V_pi_grid)