import numpy as np
import queue
from queue import PriorityQueue
from typing import Tuple
import random
import time
import itertools

#------------------------- 1A -------------------------
def init_cd(n: int)-> np.ndarray:
    '''
    Function: init_cd Date
    '''
    return np.full(n, -1, dtype=int)

def union(rep_1: int, rep_2: int, p_cd: np.ndarray)-> int:
    if(p_cd[rep_1] > p_cd[rep_2]):
        p_cd[rep_2] = rep_1
        return rep_1
    elif (p_cd[rep_2] > p_cd[rep_1]):
        p_cd[rep_1] = rep_2
        return rep_2
    else:
        p_cd[rep_2] = rep_1
        p_cd[rep_1] -= 1

    
    return p_cd

def find(ind: int, p_cd: np.ndarray)-> int:

    z = ind
    
    while p_cd[z] >= 0:
        z = p_cd[z]
    
    while p_cd[ind] >= 0:
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y
    
    return z


#------------------------- 2A -------------------------

def create_pq(n: int, l_g: list)-> queue.PriorityQueue:
    pq = PriorityQueue()

    for i in range(0, n):
        for u, v, w in l_g:
            if( i == u):
                pq.put((w, u, v))
    
    return pq

def print_pq_contents(pq: PriorityQueue):
    items = []
    while not pq.empty():
        items.append(pq.get())
    
    for item in items:
        pq.put(item)
    
    print(items)

def kruskal(n: int, l_g: list)-> Tuple[int, list]:
    weight = 0
    tree = init_cd(n)
    l_t = []


    pq = create_pq(n, l_g)

    while not pq.empty():
        (w, u, v) = pq.get()

        if(find(u, tree) != find(v, tree)):
            l_t.append((u, v))
            weight += w
            union(u, v, tree)
        
    return weight, l_t

def kruskal_2(n: int, l_g: list)-> Tuple[int, list, float]:
    weight = 0
    tree = init_cd(n)
    l_t = []
    total = 0


    pq = create_pq(n, l_g)

    while not pq.empty():
        (w, u, v) = pq.get()

        start = time.time()
        f1 = find(u, tree)
        end = time.time()
        total += end - start

        start = time.time()
        f2 = find(v, tree)
        end = time.time()
        total += end - start

        if( f1 != f2):
            l_t.append((u, v))
            weight += w
            start = time.time()
            union(u, v, tree)
            end = time.time()
            total += end - start
        
    return weight, l_t, total


#------------------------- 2B -------------------------

def complete_graph(n_nodes: int, max_weight=50)-> Tuple[int, list]:
    l_g = []

    for u in range(n_nodes):
        for v in range(u+1, n_nodes):
            w = random.randint(1, max_weight)
            l_g.append((u, v, w))

    return (n_nodes, l_g)

def time_kruskal(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int)-> list:
    l_times = []
    n_nodes = n_nodes_ini

    while n_nodes <= n_nodes_fin:
        total = 0

        for i in range(n_graphs):
            n, l_g = complete_graph(n_nodes)
            start = time.time()
            kruskal(n_nodes, l_g)
            end = time.time()
            total += end - start

        l_times.append((n, (total / n_graphs)))
        n_nodes += step

    return l_times

def time_kruskal_2(n_graphs: int, n_nodes_ini: int, n_nodes_fin: int, step: int)-> list:
    l_times = []
    n_nodes = n_nodes_ini

    while n_nodes <= n_nodes_fin:
        total = 0

        for i in range(n_graphs):
            n, l_g = complete_graph(n_nodes)
            tw, tree, time = kruskal_2(n_nodes, l_g)
            total += time

        l_times.append((n, (total / n_graphs)))
        n_nodes += step

    return l_times


#------------------------- 3A -------------------------

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

