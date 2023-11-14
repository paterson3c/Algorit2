import numpy as np
import p131A as cd
import p132A as kruskal
import p132B as time_k
from queue import PriorityQueue
from typing import Tuple
import random
import time

def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    """
    """
    m = np.random.randint(1, w_max+1, (n_nodes, n_nodes))
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0)
    return m

def greedy_tsp(dist_m: np.ndarray, node_ini=0)-> list:
    num_nodes = dist_m.shape[0]
    circuit = [node_ini]
    while len(circuit) < num_nodes:
        current_node = circuit[-1]
        # sort cities in ascending distance from current
        options = np.argsort(dist_m[ current_node ])
        # add first city in sorted list not visited yet
        for node in options:
            if node not in circuit:
                circuit.append(node)
                break
    
    return circuit + [node_ini]

def len_circuit(circuit: list, dist_m: np.ndarray)-> int:
    num_nodes = dist_m.shape[0]
    length = 0

    for j in range(num_nodes):
            length += dist_m[circuit[j]][circuit[j + 1]]

    return length
