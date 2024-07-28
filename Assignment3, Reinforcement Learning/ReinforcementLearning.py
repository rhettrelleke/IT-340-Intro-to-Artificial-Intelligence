# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:17:37 2023

@author: Rhett
"""

#Define the problem parameters
d_f = 0.8
r = -10

#Initialize state values 
state_values = {
    'A': 0,
    'B': 0,
    'C': 0,
    'D': 100  # Initial value for state D
}

#Define the state transition model p(s'|s,a)
state_transitions = {
    'A': {
        1: {'B': 0.9, 'C': 0.1},
        2: {'B': 0.1, 'C': 0.9}
    },
    'B': {
        1: {'A': 0.9, 'D': 0.1},
        2: {'A': 0.1, 'D': 0.9}
    },
    'C': {
        1: {'D': 0.9, 'A': 0.1},
        2: {'D': 0.1, 'A': 0.9}
    },
    'D': {1: {}, 2: {}}
}

    #Initialize policies 1-3
p1 = {
    'A': 1,
    'B': 2,
    'C': 2,
    'D': None #terminal state
}
p2 = {
    'A': 2,
    'B': 1,
    'C': 1,
    'D': None #terminal state
}

p3 = {
    'A': 1,
    'B': 2,
    'C': 1,
    'D': None #terminal state
}



#Define a policy evaluation 
def policy_evaluation(policy, state_values, num_iterations=100):
    for _ in range(num_iterations):
        new_state_values = state_values.copy()
        for state in state_values.keys():
            if state != 'D':  # Skip the terminal state
                action = policy[state]
                value = 0
                for next_state, prob in state_transitions[state][action].items():
                    value += prob * (r + d_f * state_values[next_state])
                new_state_values[state] = value
        state_values = new_state_values
    return state_values

#Define the policy iteration 
def policy_iteration(policy, state_values):
    policy_stable = True
    for state in policy.keys():
        if state != 'D':
            old_action = policy[state]
            best_action = None
            best_value = float('-inf')
            for action in state_transitions[state].keys():
                value = 0
                for next_state, prob in state_transitions[state][action].items():
                    value += prob * (r + d_f * state_values[next_state])
                if value > best_value:
                    best_value = value
                    best_action = action
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
    return policy, policy_stable

#Perform policy evaluation and policy iteration 
iterations = 1       
state_values = policy_evaluation(p1, state_values)
p1, policy_stable = policy_iteration(p1, state_values)
print(f"If policy {iterations} is chosen, the values of states will converge to:")
for state, value in state_values.items():
    print(f"V_π({state}) = {value:.2f}")

iterations = 2       
state_values = policy_evaluation(p2, state_values)
p2, policy_stable = policy_iteration(p2, state_values)
print(f"If policy {iterations} is chosen, the values of states will converge to:")
for state, value in state_values.items():
    print(f"V_π({state}) = {value:.2f}")

iterations = 3
state_values = policy_evaluation(p3, state_values)
p3, policy_stable = policy_iteration(p3, state_values)
print(f"If policy {iterations} is chosen, the values of states will converge to:")
for state, value in state_values.items():
    print(f"V_π({state}) = {value:.2f}")

print("Final policy:")
for state, action in p1.items():
    if state != 'D':
        print(f"π({state}) = Action {action}")

#Prove that it is equally good to visit B and C from A
B_value = state_transitions['A'][1]['B'] * (r + d_f * state_values['B']) + \
               state_transitions['A'][2]['B'] * (r + d_f * state_values['B'])

C_value = state_transitions['A'][1]['C'] * (r + d_f * state_values['C']) + \
               state_transitions['A'][2]['C'] * (r + d_f * state_values['C'])

print(f"Value from B: {B_value:.2f}")
print(f"Value from C: {C_value:.2f}")

#Comparing the visits to B and C from A
if abs(B_value - C_value) < 1e-6:
    print("It is equally good for the agent to visit state B or C from A.")
else:
    print("It is not equally good for the agent to visit state B and C from A.")




