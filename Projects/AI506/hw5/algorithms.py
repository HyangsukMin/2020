from utils import Baskets
from typing import List, Tuple

### TODO: You may import any Python's standard library here (Do not import other external libraries) ###
from itertools import combinations
### Import End ###

### TODO: You may declare any additional function here if needed ###

### Import End ###

def naive_with_matrix(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the naive algorithm with matrix
    results = []
    # Create empty upper triangular matrix
    mat = []
    m = 10000 # Randomly setting
    n = 10000 # Randomly setting
    for _ in range(m):
        mat.append([0]*(n-1))
        n -= 1
    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        for i,j in combinations(basket,2):
            mat[(i)][(j-i-1)] += 1
    for i in range(m):
        results.extend([(i,idx+i+1,element) for idx, element in enumerate(mat[i]) if element >= threshold])
    results = sorted(results,key = lambda x: (x[0],x[1],x[2]))
    return results

def naive_with_triples(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the naive algorithm with triples
    mat = {}
    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        for i,j in combinations(basket,2):
            if (i,j) in mat.keys():
                mat[(i,j)] += 1
            else :
                mat[(i,j)] = 1
    results = [(i,j,value) for (i,j), value in mat.items() if value > threshold]
    results = sorted(results,key = lambda x: (x[0],x[1],x[2]))
    return results

def apriori_with_matrix(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the apriori algorithm with matrix
    results = []
    # Pass1
    ## Find Frequent Items
    # Randomly set the maximum number
    frequent_items = [0]*10000

    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        for idx in basket:
            frequent_items[idx] += 1

    frequent_items = [idx for idx, x in enumerate(frequent_items) if x > threshold]
    frequent_items.sort()
    
    # Pass2
    ## Create empty upper triangular matrix
    mat = []
    m = len(frequent_items)
    n = len(frequent_items)

    for _ in range(m):
        mat.append([0]*(n-1))
        n -= 1

    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        basket = [frequent_items.index(x) for x in basket if x in frequent_items]
        for i,j in combinations(basket,2):
            mat[i][(j-i-1)] += 1
     
    for i in range(m):
        results.extend([(frequent_items[i],frequent_items[idx+i+1],element) for idx, element in enumerate(mat[i]) if element >= threshold])
    results = sorted(results,key = lambda x: (x[0],x[1],x[2]))
    return results

def apriori_with_triples(baskets: Baskets, threshold: int) -> List[Tuple[int, int, int]]:
    # TODO: Implement the apriori algorithm with triples 
    # Pass1
    ## Find Frequent Items
    # Randomly set the maximum number
    frequent_items = [0]*10000
    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        for idx in basket:
            frequent_items[idx] += 1

    frequent_items = [idx for idx, x in enumerate(frequent_items) if x > threshold]
    frequent_items.sort()

    results = []
    mat = {}
    while True:
        basket = baskets.readItems()
        if basket == None:
            break
        basket = [frequent_items.index(x) for x in basket if x in frequent_items]
        for i,j in combinations(basket,2):
            if (i,j) in mat.keys():
                mat[(i,j)] += 1
            else :
                mat[(i,j)] = 1
    results = [(frequent_items[i],frequent_items[j],value) for (i,j), value in mat.items() if value > threshold]
    results = sorted(results,key = lambda x: (x[0],x[1],x[2]))  
    return results