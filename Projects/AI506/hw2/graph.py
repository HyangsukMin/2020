from collections import defaultdict
from typing import DefaultDict, Dict, Set, TextIO, Tuple, Union

### You may import any Python's standard library here (Do not import other external libraries) ###

### Import End ###


class Graph:
    _nodes: Set[int] = set()  # Set of nodes
    _in_degree: Dict[int, int] = dict()  # key: node, value: in-degree of node
    _out_degree: Dict[int, int] = dict()  # key: node, value: out-degree of node

    @property
    def nodes(self) -> Set[int]:
        return self._nodes

    @property
    def in_degree(self) -> Dict[int, int]:
        return self._in_degree

    @property
    def out_degree(self) -> Dict[int, int]:
        return self._out_degree


# Memory-based Graph
class Graph_memory(Graph):
    def __init__(self, filePath: str) -> None:
        ################## TODO: Fill out the class variables ##################
        #  self._nodes (parent)                                                #
        #  self._in_degree (parent)                                            #
        #  self._out_degree (parent)                                           #
        #  self.__out_neighbor :  key: node, value: out-neighbor nodes         #
        #                                                                      #
        # WARNING: Do not declare another class variables                      #
        ########################################################################

        self._nodes = set()
        self._in_degree = dict()
        self._out_degree = dict()
        self.__out_neighbor: Dict[int, Set[int]] = dict()

        with open(filePath) as f:
            for line in f.readlines():
                u,v = map(int,line.rstrip().split('\t')) # parse src, dst
                self._nodes.add(u) # add node to the set
                self._nodes.add(v)
                # count in_degree for each node
                if v not in self._in_degree.keys():
                    self._in_degree[v] = 1
                else :
                    self._in_degree[v] += 1
                # count out_degree for each node
                if u not in self._out_degree.keys():
                    self._out_degree[u] = 1
                else :
                    self._out_degree[u] += 1
                
                # store edge information for each node
                if u not in self.__out_neighbor.keys():
                    self.__out_neighbor[u] = {v}
                else:
                    self.__out_neighbor[u].add(v)
        ######################### Implementation end ###########################

    @property
    def out_neighbor(self) -> Dict[int, Set[int]]:
        return self.__out_neighbor


# Disk-based Graph
class Graph_disk(Graph):
    def __init__(self, filePath: str) -> None:
        self.__f: TextIO = open(filePath, "r")  # File object

        ################## TODO: Fill out the class variables ##################
        #  self._nodes (parent)                                                #
        #  self._in_degree (parent)                                            #
        #  self._out_degree (parent)                                           #
        #                                                                      #
        # WARNING: Do not declare another class variables                      #
        ########################################################################

        self._nodes = set()
        self._in_degree = dict()
        self._out_degree = dict()

        for line in self.__f.readlines():
            u,v = map(int,line.rstrip().split('\t')) # parse src, dst
            self._nodes.add(u) # add node to the set
            self._nodes.add(v)

            # count in_degree for each node
            if v not in self._in_degree.keys():
                self._in_degree[v] = 1
            else :
                self._in_degree[v] += 1
            
            # count out_degree for each node
            if u not in self._out_degree.keys():
                self._out_degree[u] = 1
            else :
                self._out_degree[u] += 1
        ######################### Implementation end ###########################

    def __del__(self) -> None:
        self.__f.close()

    # Read one edge from the file
    def readEdge(self) -> Union[Tuple[int, int], None]:
        #################### TODO: Complete the function #######################
        # Read one line from the file object                                   #
        # If you cannot read it, then return None                              #
        # Otherwise, parse the line to find the source and destination nodes   #
        # Then, return the tuple (src, dst) where src, dst are integers        #
        ########################################################################
        edge = self.__f.readline()
        if edge == '' :
            return None
        else :
            edge = tuple(map(int,edge.rstrip().split('\t')))
            return edge
        ######################### Implementation end ###########################

    # Move the file pointer into the beginning of the file
    def setBOF(self) -> None:
        self.__f.seek(0)
