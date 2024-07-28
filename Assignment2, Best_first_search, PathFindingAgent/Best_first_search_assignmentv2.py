#Rhett Relleke
# Define the map

Romania_Map = {'Oradea': {'Zerind': 71, 'Sibiu': 151},
               'Zerind': {'Arad': 75, 'Oradea': 71},
               'Arad': {'Zerind': 75, 'Sibiu': 140, 'Timisoara': 118},
               'Timisoara': {'Arad': 118, 'Lugoj': 111},
               'Lugoj': {'Timisoara': 111, 'Mehadia': 70},
               'Mehadia': {'Lugoj': 70, 'Drobeta': 75},
               'Drobeta': {'Mehadia': 75, 'Craiova': 120},
               'Craiova': {'Drobeta': 120, 'Rimnicu Vilcea': 146, 'Pitesti': 138},
               'Rimnicu Vilcea': {'Craiova': 146, 'Sibiu': 80, 'Pitesti': 97},
               'Sibiu': {'Oradea': 151, 'Arad': 140, 'Fagaras': 99, 'Rimnicu Vilcea': 80},
               'Fagaras': {'Sibiu': 99, 'Bucharest': 211},
               'Pitesti': {'Rimnicu Vilcea': 97, 'Craiova': 138, 'Bucharest': 101},
               'Bucharest': {'Fagaras': 211, 'Pitesti': 101, 'Giurgiu': 90, 'Urziceni': 85},
               'Giurgiu': {'Bucharest': 90},
               'Urziceni': {'Bucharest': 85, 'Vaslui': 142, 'Hirsova': 98},
               'Neamt': {'Iasi': 87},
               'Iasi': {'Neamt': 87, 'Vaslui': 92},
               'Vaslui': {'Iasi': 92, 'Urziceni': 142},
               'Hirsova': {'Urziceni': 98, 'Eforie': 86},
               'Eforie': {'Hirsova': 86}
               }

# Define the node class

class Node:
    def __init__(self, name, parent=None, path_cost=0, map=Romania_Map):
        self.name = name
        self.parent = parent
        self.map = map
        self.path_cost = path_cost
    
    def __lt__(self, other):
        # Compare nodes based on their path_cost
        return self.path_cost < other.path_cost

from heapq import heappop as pop
from heapq import heappush as push

# Define the two functions from the pseudo code

def Best_First_Search(problem, f):
    node = Node(problem.initial)
    frontier = [(f(node), node)]  # Priority queue ordered by f
    reached = {problem.initial: node}  # Lookup table with one entry
    while frontier:
        _, node = pop(frontier)
        if problem.Is_Goal(node.name):
            path = []
            cost = node.path_cost
            while node:
                path.insert(0, node.name)
                node = node.parent
            return path, cost
        for child in Expand(problem, node):
            s = child.name
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                push(frontier, (f(child), child))
    return "Failure"

def Expand(problem, node):
    s = node.name
    for action in problem.Actions(s):
        s_prime = problem.Result(s, action)
        cost = node.path_cost + problem.Action_Cost(s, action, s_prime)
        yield Node(s_prime, node, cost)

# Define the problem-specific functions

class Problem:
    def __init__(self, initial, goal, map=Romania_Map):
        self.initial = initial
        self.goal = goal
        self.map = map

    def Is_Goal(self, state):
        return state == self.goal

    def Actions(self, state):
        if state in self.map:
            return list(self.map[state].keys())
        return []

    def Result(self, state, action):
        return action

    def Action_Cost(self, state, action, next_state):
        if state in self.map and action in self.map[state]:
            return self.map[state][action]
        return float('inf')

# Test cases

if __name__ == "__main__":
    problem = Problem('Arad', 'Bucharest')
    path, cost = Best_First_Search(problem, lambda node: problem.Action_Cost(node.name, node.name, node.name))
    print(path, cost)  # Expected output: ['Arad', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest'] 418

    # Additional test cases
    test_cases = [
        ('Oradea', 'Arad', ['Oradea', 'Zerind', 'Arad'], 146),
        ('Oradea', 'Bucharest', ['Oradea', 'Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest'], 418),
        ('Neamt', 'Fagaras', "Failure", "Failure"),
        ('Craiova', 'Urziceni', ['Craiova', 'Pitesti', 'Bucharest', 'Urziceni'], 418),
        ('Hirsova', 'Eforie', ['Hirsova', 'Eforie'], 86)
    ]

    for start, end, expected_path, expected_cost in test_cases:
        problem = Problem(start, end)
        path, cost = Best_First_Search(problem, lambda node: problem.Action_Cost(node.name, node.name, node.name))
        print(f'Path: {path if path != "Failure" else "Failure"} {cost if path != "Failure" else "Failure"}')
        print('=' * 90)
