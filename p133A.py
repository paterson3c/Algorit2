import numpy as np
import p131A as cd
import p132A as kruskal
import p132B as time_k
from queue import PriorityQueue
from typing import Tuple
import random
import time
import itertools

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


def repeated_greedy_tsp(dist_m: np.ndarray)-> list:
    """
    aplicar nuestra funcion greedy_tsp a partir de todos los nodos del grafo y devolver el circuito con la menor longitud
    """
    num_nodes = dist_m.shape[0]
    min_length = float('inf')
    min_circuit = []
    for i in range(num_nodes):
        circuit = greedy_tsp(dist_m, i)
        length = len_circuit(circuit, dist_m)
        if length < min_length:
            min_length = length
            min_circuit = circuit
    
    return min_circuit

def exhaustive_tsp(dist_m: np.ndarray)-> list:
    """
    Para grafos pequenos podemos intentar resolver TSP simplemente examinando todos los posibles
    circuitos y devolviendo aquel con la distancia mas corta. Escribir una funcion
    exhaustive_tsp(dist_m: np.ndarray)-> List
    que implemente esta idea usando la libreria itertools . Entre los metodos de iteracion implementados en la biblioteca,
    se encuentra la funcion permutations(iterable, r=None) que devuelve un objeto iterable que proporciona sucesivamente
    todas las permutaciones de longitud r en orden lexicografico. Aqui r es por defecto la longitud del iterable pasado
    como parametro, es decir, se generan todas las permutaciones con len(iterable) elementos
    """
    num_nodes = dist_m.shape[0]
    min_length = float('inf')
    min_circuit = []
    for circuit in itertools.permutations(range(num_nodes)):
        circuit = list(circuit)
        circuit.append(circuit[0])
        length = len_circuit(circuit, dist_m)
        if length < min_length:
            min_length = length
            min_circuit = circuit
    
    return min_circuit