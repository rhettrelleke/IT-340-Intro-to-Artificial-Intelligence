# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 17:57:55 2023

@author: Rhett
"""

# Define the problem parameters
discounting_factor = 0.8
reward = -10
initial_values = {'A': 0, 
                  'B': 0, 
                  'C': 0, 
                  'D': 100}
num_iterations = 100

# Define the policy for each state
policy1 = {'A': {'1': 1, '2': 0}, 'B': {'1': 1, '2': 0}, 'C': {'1': 1, '2': 0}}
policy2 = {'A': {'1': 0, '2': 1}, 'B': {'1': 0, '2': 1}, 'C': {'1': 0, '2': 1}}
policy3 = {'A': {'1': 0.4, '2': 0.6}, 'B': {'1': 1, '2': 0}, 'C': {'1': 0, '2': 1}}

#define the state transition model
state_transitions = {
    'A': {'1': {'B': 0.9, 'C': 0.1}, 
          '2': {'B': 0.1, 'C': 0.9}},
   'B': {'1': {'A': 0.9, 'D': 0.1}, 
         '2': {'A': 0.1, 'D': 0.9}},
   'C': {'1': {'D': 0.9, 'A': 0.1}, 
         '2': {'A': 0.1, 'D': 0.9}
         },
   'D': {1: {}, 2: {}}
}

def policy_evaluation(policy, state_transitions, initial_values, num_iterations):
    state_values = initial_values.copy()
    for _ in range(num_iterations):
        new_state_values = initial_values.copy()
        for state in state_values.keys():
            if state != 'D':
                action = policy[state]
                v_s = 0
                for next_state, transition_prob in state_transitions[state][action].items():
                    v_s += transition_prob * (reward + discounting_factor * state_values[next_state])
                new_state_values[state] = v_s
        state_values = new_state_values
    return state_values

def policy_improvement(policy, state_values):
    policy_stable = True
    for state in policy.keys():
        if state != 'D':
            old_action = policy[state]
            best_action = None
            best_val = float('-inf')
            for action in state_transitions[state].keys():
                value = 0
                for next_state, transition_prob in state_transitions[state][action].items():
                    value += transition_prob * (reward + discounting_factor * state_values[next_state])
                if value > best_val:
                    best_val = value
                    best_action = action
            policy[state] = best_action
            if old_action != best_action:
                policy_stable = False
    return policy, policy_stable

policy_stable = False
iterations = 0

while not policy_stable:
    state_values = policy_evaluation(policy, state_transitions, initial_values, num_iterations)
    policy, policy_stable = policy_improvement(policy, state_values)
    iterations += 1
    

# Perform policy evaluation for policy 1
result_policy1 = policy_evaluation(policy1, state_transitions, discounting_factor, initial_values, num_iterations)

# Perform policy evaluation for policy 2
result_policy2 = policy_evaluation(policy2, state_transitions, discounting_factor, initial_values, num_iterations)

# Perform policy evaluation for policy 3
result_policy3 = policy_evaluation(policy3, state_transitions, discounting_factor, initial_values, num_iterations)

# Print the results
print("Policy 1 State Values:")
for state, value in result_policy1.items():
    print(f'v_π({state}) = {value:.2f}')

print("\nPolicy 2 State Values:")
for state, value in result_policy2.items():
    print(f'v_π({state}) = {value:.2f}')

print("\nPolicy 3 State Values:")
for state, value in result_policy3.items():
    print(f'v_π({state}) = {value:.2f}')


