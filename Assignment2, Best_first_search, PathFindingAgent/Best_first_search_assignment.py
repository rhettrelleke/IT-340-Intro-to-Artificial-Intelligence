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


from heapq import heappop as pop
from heapq import heappush as push

class Problem:
        def __init__(self, initial, goal):
            self.initial = initial
            self.goal = goal
        
        def Is_Goal(self, state):
            return state == self.goal
        
        def Actions(self, state):
            return list(Romania_Map[state].keys())
        
        def Result(self, action):
            if action in Romania_Map.get(Node.name, {}):
                return action
            return None
        
        def Action_Cost(self, state, action, state_prime):
            return Romania_Map[state][state_prime]


class Node:
    def __init__(self, name, parent=None, path_cost=0, map=Romania_Map):
        self.name = name
        self.parent = parent
        self.map = map
        self.path_cost = path_cost
#Todos:
#1. Define the two functions from the pseudo code
def Best_First_Search(problem, f):
    node = Node(problem.initial)
    frontier = [(f(node), node)]  # Priority queue ordered by f
    reached = {problem.initial: node}  # Lookup table

    while frontier:
        _, node = pop(frontier)
        if problem.Is_Goal(node.name):
            return node
        for child in Expand(problem, node):
            s = child.name
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                push(frontier, (f(child), child))
    return None

def Expand(problem, node):
    s = node.name
    for action in problem.Actions(s):
        s_prime = action
        cost = node.path_cost + problem.Action_Cost(s, action, s_prime)
        yield Node(name=s_prime, parent=node, path_cost=cost)

    

#2. Provide, at least, 5 test cases
def test_Best_First_Search():
    def h(node):
        return heuristic_function(node)  # You can define your heuristic function here

    def heuristic_function(node, goal):
    # Calculate the heuristic cost from the current node to the goal node
    # For example, you can use a simple lookup in the Romania_Map
        if node.name in Romania_Map and goal in Romania_Map[node.name]:
            return Romania_Map[node.name][goal]
        else:
            return float('inf')  # Return a large value if there's no direct connection
    
    
    # Test case 1: Starting from 'Arad' to 'Bucharest'
    problem1 = Problem('Arad', 'Bucharest')
    solution_node = Best_First_Search(problem1, heuristic_function)
    if solution_node:
        path = []
        while solution_node:
            path.insert(0, solution_node.name)
            solution_node = solution_node.parent
        assert path == ['Sibiu', 'Rimnicu Vilcea', 'Pitesti', 'Bucharest']
        assert solution_node.path_cost == 418
    else:
        assert False

test_Best_First_Search()