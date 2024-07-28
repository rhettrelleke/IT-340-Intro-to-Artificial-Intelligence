# -*- coding: utf-8 -*-
"""
Created on Sun Oct 15 15:17:37 2023

@author: Rhett
"""

def round_dictionary_key(dictionary, key):
    dictionary[key] = round(dictionary[key], 2)


def policy_iteration():
    r = -10  # reward
    d_f = 0.8  #discounting factor

    # Initialize the value function and policy
    state_val = {
        "V(A)": 0,
        "V(B)": 0,
        "V(C)": 0,
        "V(D)": 100
    }
    
    # Initialize the policy with arbitrary actions for each state
    pi = {
        "V(A)": "B",
        "V(B)": "C",
        "V(C)": "A",
    }

    policy_stable = False

    while not policy_stable:
        # Policy Evaluation
        for _ in range(100):
            delta = 0
            for state in state_val:
                if state != "V(D)":
                    old_state = state_val[state]
                    state_val[state] = 0
                    action = pi[state]  # Use the state name as the key to get the action
                    state_val[state] = 0.9 * (r + d_f * state_val[action]) + 0.1 * (
                            r + d_f * state_val["V(D)"])
                    delta = max(delta, abs(old_state - state_val[state]))

            # Check for convergence
            if delta < 1e-6:
                break

        # Policy Improvement
        policy_stable = True
        for state in state_val:
            if state != "V(D)":
                old_action = pi[state]
                max_action = None
                max_value = float("-inf")
                for action in state_val:
                    action_value = 0.9 * (r + d_f * state_val[action]) + 0.1 * (
                            r + d_f * state_val["V(D)"])
                    if action_value > max_value:
                        max_value = action_value
                        max_action = action
                pi[state] = max_action
                if old_action != pi[state]:
                    policy_stable = False

    # Round the value to two decimal places
    for state in state_val:
        state_val[state] = round(state_val[state], 2)

    return state_val, pi




def main():
    r = -10  # r
    d_f = 0.8 #discounting factor

    original_state_value_table = {
        "V(A)": 0,
        "V(B)": 0,
        "V(C)": 0,
        "V(D)": 100
    }

    # If Policy 1 is chosen
    policy1_state_value_table = original_state_value_table.copy()

    for i in range(0, 100):
        policy1_state_value_table["V(B)"] = 0.9 * (
                r + (d_f * policy1_state_value_table["V(D)"])) + 0.1 * (
                r + (d_f * policy1_state_value_table["V(A)"]))
        policy1_state_value_table["V(C)"] = 0.9 * (
                r + (d_f * policy1_state_value_table["V(A)"])) + 0.1 * (
                r + (d_f * policy1_state_value_table["V(D)"]))
        policy1_state_value_table["V(A)"] = 0.9 * (
                r + (d_f * policy1_state_value_table["V(B)"])) + 0.1 * (
                r + (d_f * policy1_state_value_table["V(C)"]))

    # Round the value to two decimal places
    round_dictionary_key(policy1_state_value_table, "V(A)")
    round_dictionary_key(policy1_state_value_table, "V(B)")
    round_dictionary_key(policy1_state_value_table, "V(C)")

    # Print the value of states
    print("If policy 1 is chosen, the values of states will converge to: ")
    print(policy1_state_value_table)
    print()

    # If Policy 2 is chosen
    # If Policy 2 is chosen
    policy2_state_value_table = original_state_value_table.copy()
    
    for i in range(0, 100):
        policy2_state_value_table["V(C)"] = 0.9 * (
                r + (d_f * policy2_state_value_table["V(D)"])) + 0.1 * (
                r + (d_f * policy2_state_value_table["V(A)"]))
        policy2_state_value_table["V(B)"] = 0.9 * (
                r + (d_f * policy2_state_value_table["V(A)"])) + 0.1 * (
                r + (d_f * policy2_state_value_table["V(D)"]))
        policy2_state_value_table["V(A)"] = 0.9 * (
                r + (d_f * policy2_state_value_table["V(C)"])) + 0.1 * (
                r + (d_f * policy2_state_value_table["V(B)"]))
    
    # Round the value to two decimal places
    round_dictionary_key(policy2_state_value_table, "V(A)")
    round_dictionary_key(policy2_state_value_table, "V(B)")
    round_dictionary_key(policy2_state_value_table, "V(C)")
    
    # Print the value of states
    print("If policy 2 is chosen, the values of states will converge to: ")
    print(policy2_state_value_table)
    print()


    # If Policy 3 is chosen
    policy3_state_value_table = original_state_value_table.copy()

    for i in range(0, 100):
        policy3_state_value_table["V(A)"] = 0.9 * (
                r + (d_f * policy3_state_value_table["V(B)"])) + 0.1 * (
                r + (d_f * policy3_state_value_table["V(C)"]))
        policy3_state_value_table["V(B)"] = 0.9 * (
                r + (d_f * policy3_state_value_table["V(D)"])) + 0.1 * (
                r + (d_f * policy3_state_value_table["V(A)"]))
        policy3_state_value_table["V(C)"] = 0.9 * (
                r + (d_f * policy3_state_value_table["V(D)"])) + 0.1 * (
                r + (d_f * policy3_state_value_table["V(A)"]))

    round_dictionary_key(policy3_state_value_table, "V(A)")
    round_dictionary_key(policy3_state_value_table, "V(B)")
    round_dictionary_key(policy3_state_value_table, "V(C)")

    print("If policy 3 is chosen, the values of states will converge to: ")
    print(policy3_state_value_table)

    # Policy Iteration
    final_v, final_pi = policy_iteration()

    print("Final state values obtained from Policy Iteration:")
    print(final_v)
    print()

    print("Final policy obtained from Policy Iteration:")
    print(final_pi)
    print()

    # Check if visiting B and C from A is equally good
    v_from_A_to_B = final_v["V(B)"]
    v_from_A_to_C = final_v["V(C)"]

    print("Value of visiting B from A:", v_from_A_to_B)
    print("Value of visiting C from A:", v_from_A_to_C)

if __name__ == "__main__":
    main()


