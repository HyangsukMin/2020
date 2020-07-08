from collections import defaultdict
from typing import DefaultDict, Dict, List, TextIO
from graph import Graph_disk, Graph_memory

### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###


# Compute the distance between two dictionaries based on L1 norm
def l1_distance(x: DefaultDict[int, float], y: DefaultDict[int, float]) -> float:
    err: float = 0.0
    for k in x.keys():
        err += abs(x[k] - y[k])
    return err


################################################################################
# Run the pagerank algorithm iteratively using the memory-based graph          #
#  parameters                                                                  #
#    - graph : Memory-based graph (Graph_memory object)                        #
#    - damping_factor : Damping factor                                         #
#    - preference : Preference vector                                          #
#    - maxiters : The maximum number of iterations                             #
#    - tol : Tolerance threshold to check the convergence                      #
################################################################################
def pagerank_memory(
    graph: Graph_memory,
    damping_factor: float,
    preference: Dict[int, float],
    maxiters: int,
    tol: float,
) -> Dict[int, float]:
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector
    ############### TODO: Implement the pagerank algorithm #####################

    ### Create the Initial pagerank vector
    old_vec = dict().fromkeys(graph.nodes,1/len(graph.nodes)) 
    
    for itr in range(maxiters):
        ### Make new pagerank vector for every iter
        vec: DefaultDict[int, float] = defaultdict(float)

        ### start calculating pagerank for each page i
        for i in graph.nodes: 
#            ### If in_degree of page i is 0, make pagerank = 0
#            if i not in graph.in_degree.keys(): 
#                vec[i] = 0

            ### If out_degree of page i is 0, pass it.
            if i not in graph.out_degree.keys():
                continue
            
            ### Update pagerank for pages
            for o in graph.out_neighbor[i]:
                vec[o] += damping_factor*(old_vec[i]/graph.out_degree[i])
        
        ### Add probability that Jumps to some random page.
        for i in vec.keys():
            vec[i] += (1-damping_factor)*preference[i]
        #### Check the convergence ###
        # Stop the iteration if L1norm[PR(t) - PR(t-1)] < tol
        delta: float = l1_distance(old_vec,vec)
        print(f"[Iter {itr}]\tDelta = {delta}")
        if delta < tol:
            break
        old_vec = vec.copy()
    ########################### Implementation end #############################
    return dict(vec)


################################################################################
# Run the pagerank algorithm iteratively using the disk-based graph            #
#  parameters                                                                  #
#    - graph : Disk-based graph (Graph_disk) object                            #
#    - damping_factor : Damping factor                                         #
#    - preference : Preference vector                                          #
#    - maxiters : The maximum number of iterations                             #
#    - tol : Tolerance threshold to check the convergence                      #
################################################################################
def pagerank_disk(
    graph: Graph_disk,
    damping_factor: float,
    preference: Dict[int, float],
    maxiters: int,
    tol: float,
) -> Dict[int, float]:
    vec: DefaultDict[int, float] = defaultdict(float)  # Pagerank vector
    ############### TODO: Implement the pagerank algorithm #####################

    ### Create the Initial pagerank vector
    old_vec = dict().fromkeys(graph.nodes,1/len(graph.nodes))
    for itr in range(maxiters):
        ### Make new pagerank vector for every iter
        vec: DefaultDict[int, float] = defaultdict(float)

        ### Move the file pointer into the beginning of the file
        graph.setBOF()
        
        ### Execute until reach at the end of the file.
        while True:
            edge = graph.readEdge()
            if edge == None:
                break
            else :
                ### Update pagerank for pages
                i,o = edge
                vec[o] += damping_factor*(old_vec[i] / graph.out_degree[i])

        ### Add probability that Jumps to some random page.
        for i in vec.keys():
            vec[i] += (1-damping_factor)*preference[i]
        
        #### Check the convergence ###
        # Stop the iteration if L1norm[PR(t) - PR(t-1)] < tol
        delta: float = l1_distance(vec,old_vec)
        print(f"[Iter {itr}]\tDelta = {delta}")
        if delta < tol:
            break
        
        old_vec = vec.copy()
    ########################### Implementation end #############################

    return dict(vec)
