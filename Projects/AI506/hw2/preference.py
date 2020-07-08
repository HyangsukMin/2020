from typing import Dict
from graph import Graph

### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###


def preference_uniform(graph: Graph) -> Dict[int, float]:
    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) should be the constant                          #
    ############################################################################

    preference: Dict[int, float] = dict().fromkeys(graph.nodes,1/len(graph.nodes))

    ########################### Implementation end #############################

    return preference


def preference_onehot(graph: Graph, node: int) -> Dict[int, float]:
    assert node in graph.nodes

    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) = | 1   if i is given as a parameter            #
    #                            | 0   otherwise                               #
    ############################################################################

    preference: Dict[int, float] = dict().fromkeys(graph.nodes,0)
    preference[node] = 1
    
    ########################### Implementation end #############################

    return preference


def preference_degree(graph: Graph) -> Dict[int, float]:
    ################## TODO: Fill out the preference vector ####################
    #  Prob(node i is choosen) ∝ 1 + in-degree(i)                             #
    ############################################################################

    preference: Dict[int, float] = dict().fromkeys(graph.nodes,0) # initialize with 0 for all nodes
    preference.update(graph.in_degree) # if node has in_degree, update it
    d = sum(preference.values()) + len(preference.keys()) # sum in-degree of all nodes 
    m = [(x+1)/d for x in preference.values()]
    preference = dict(zip(preference.keys(),m))

    ########################### Implementation end #############################

    return preference
