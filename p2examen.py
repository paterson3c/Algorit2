from typing import Tuple
import random
import time
import itertools
import queue
from queue import PriorityQueue
import numpy as np

# ------------------------- 1A -------------------------

def init_cd(n: int) -> np.ndarray:
    # Inicializa un conjunto disjunto (Union-Find) para 'n' elementos.
    # Cada elemento se inicializa como su propio representante.
    # Se usa un array de NumPy para el almacenamiento.
    return np.full(n, -1, dtype=int)

def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    # Realiza la unión de dos subconjuntos en un conjunto disjunto.
    # Combina los conjuntos representados por rep_1 y rep_2.
    # Actualiza el array p_cd para reflejar la nueva estructura del conjunto disjunto.
    if p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    if p_cd[rep_2] > p_cd[rep_1]:
        p_cd[rep_1] = rep_2
        return rep_2
    p_cd[rep_2] = rep_1
    p_cd[rep_1] -= 1
    return rep_1

def find(ind: int, p_cd: np.ndarray) -> int:
    # Encuentra el representante del subconjunto al que pertenece el elemento ind.
    # Realiza una compresión de caminos para mejorar la eficiencia en búsquedas futuras.
    z = ind
    while p_cd[z] >= 0:  # Encuentra el representante
        z = p_cd[z]
    while p_cd[ind] >= 0:  # Comprime el camino
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y
    return z

# ------------------------- 2A -------------------------

def create_pq(n: int, l_g: list) -> queue.PriorityQueue:
    # Crea una cola de prioridad para los bordes de un grafo.
    # Los bordes se insertan en la cola con su peso como clave de ordenación.
    pq = PriorityQueue()
    for i in range(0, n):
        for u, v, w in l_g:
            if i == u:
                pq.put((w, u, v))
    return pq

def kruskal(n: int, l_g: list) -> Tuple[int, list]:
    # Implementa el algoritmo de Kruskal para encontrar el árbol de expansión mínima de un grafo.
    # Utiliza un conjunto disjunto y una cola de prioridad para determinar los bordes a incluir.
    weight = 0  # Peso total del árbol de expansión mínima
    tree = init_cd(n)  # Inicializa el conjunto disjunto
    l_t = []  # Lista para almacenar los bordes del árbol
    pq = create_pq(n, l_g)  # Crea la cola de prioridad con los bordes
    while not pq.empty():
        (w, u, v) = pq.get()
        if find(u, tree) != find(v, tree):  # Verifica si los nodos están en diferentes subconjuntos
            l_t.append((u, v))
            weight += w
            union(u, v, tree)
    return weight, l_t

def kruskal_2(n: int, l_g: list) -> Tuple[int, list, float]:
    # Versión modificada del algoritmo de Kruskal que también mide el tiempo de ejecución.
    # Similar a kruskal, pero añade mediciones de tiempo para las operaciones de find y union.
    weight = 0
    tree = init_cd(n)
    l_t = []
    total = 0
    pq = create_pq(n, l_g)
    while not pq.empty():
        (w, u, v) = pq.get()
        start = time.time()  # Inicia la medición de tiempo
        f1 = find(u, tree)
        end = time.time()  # Finaliza la medición de tiempo
        total += end - start
        start = time.time()
        f2 = find(v, tree)
        end = time.time()
        total += end - start
        if f1 != f2:
            l_t.append((u, v))
            weight += w
            start = time.time()
            union(u, v, tree)
            end = time.time()
            total += end - start
    return weight, l_t, total

# ------------------------- 2B -------------------------

def complete_graph(n_nodes: int, max_weight=50) -> Tuple[int, list]:
    # Genera un grafo completo con pesos aleatorios en los bordes.
    # Cada par de nodos tiene un borde con un peso asignado aleatoriamente.
    l_g = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            w = random.randint(1, max_weight)  # Genera un peso aleatorio
            l_g.append((u, v, w))
    return (n_nodes, l_g)

def time_kruskal(
        n_graphs: int,
        n_nodes_ini: int,
        n_nodes_fin: int,
        step: int) -> list:
    # Mide el tiempo promedio de ejecución del algoritmo de Kruskal en una serie de grafos.
    # Genera grafos con un número creciente de nodos y calcula el tiempo promedio de ejecución de Kruskal.
    l_times = []
    n_nodes = n_nodes_ini
    while n_nodes <= n_nodes_fin:
        total = 0
        for _ in range(n_graphs):
            n, l_g = complete_graph(n_nodes)
            start = time.time()
            kruskal(n_nodes, l_g)
            end = time.time()
            total += end - start
        l_times.append((n, (total / n_graphs)))
        n_nodes += step
    return l_times

def time_kruskal_2(
        n_graphs: int,
        n_nodes_ini: int,
        n_nodes_fin: int,
        step: int) -> list:
    # Similar a time_kruskal, pero utiliza la función kruskal_2 que mide el tiempo de las operaciones internas.
    l_times = []
    n_nodes = n_nodes_ini
    while n_nodes <= n_nodes_fin:
        total = 0
        for _ in range(n_graphs):
            n, l_g = complete_graph(n_nodes)
            time = kruskal_2(n_nodes, l_g)[2]
            total += time
        l_times.append((n, (total / n_graphs)))
        n_nodes += step
    return l_times

# ------------------------- 3A -------------------------

def dist_matrix(n_nodes: int, w_max=10) -> np.ndarray:
    # Genera una matriz de distancias simétrica para un grafo completo.
    # Cada elemento de la matriz representa la distancia entre dos nodos.
    m = np.random.randint(1, w_max + 1, (n_nodes, n_nodes))
    m = (m + m.T) // 2  # Asegura que la matriz sea simétrica
    np.fill_diagonal(m, 0)  # Pone 0s en la diagonal (distancia de un nodo a sí mismo)
    return m

def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    # Implementa un algoritmo greedy para el problema del viajante de comercio (TSP).
    # Comienza en un nodo inicial y siempre visita el nodo no visitado más cercano.
    num_nodes = dist_m.shape[0]
    circuit = [node_ini]  # Inicia el circuito en el nodo inicial
    while len(circuit) < num_nodes:
        current_node = circuit[-1]
        options = np.argsort(dist_m[current_node])  # Ordena los nodos por distancia
        for node in options:  # Elige el nodo más cercano no visitado
            if node not in circuit:
                circuit.append(node)
                break
    return circuit + [node_ini]  # Cierra el circuito volviendo al nodo inicial

def len_circuit(circuit: list, dist_m: np.ndarray) -> int:
    # Calcula la longitud total de un circuito dado en el TSP.
    # Suma las distancias entre nodos consecutivos en el circuito.
    num_nodes = dist_m.shape[0]
    length = 0
    for j in range(num_nodes):
        length += dist_m[circuit[j]][circuit[j + 1]]
    return length

def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    # Aplica el algoritmo greedy para el TSP desde cada nodo y elige el mejor circuito.
    # Busca encontrar una solución aproximada al TSP desde diferentes puntos de inicio.
    num_nodes = dist_m.shape[0]  # Número total de nodos en el grafo
    min_length = float('inf')  # Inicializa la longitud mínima a infinito
    min_circuit = []  # Inicializa el circuito mínimo vacío

    # Itera sobre cada nodo del grafo como punto de inicio
    for i in range(num_nodes):
        circuit = greedy_tsp(dist_m, i)  # Obtiene un circuito empezando en el nodo i
        length = len_circuit(circuit, dist_m)  # Calcula la longitud del circuito
        # Actualiza el circuito mínimo si se encuentra uno más corto
        if length < min_length:
            min_length = length
            min_circuit = circuit

    return min_circuit  # Devuelve el circuito más corto encontrado

def exhaustive_tsp(dist_m: np.ndarray) -> list:
    # Resuelve el TSP mediante un enfoque exhaustivo, probando todas las permutaciones de nodos.
    # Encuentra la solución óptima, pero es impráctico para grafos grandes debido a su complejidad factorial.
    num_nodes = dist_m.shape[0]  # Número total de nodos en el grafo
    min_length = float('inf')  # Inicializa la longitud mínima a infinito
    min_circuit = []  # Inicializa el circuito mínimo vacío

    # Itera sobre todas las permutaciones posibles de los nodos
    for circuit in itertools.permutations(range(num_nodes)):
        circuit = list(circuit) + [circuit[0]]  # Cierra el circuito volviendo al nodo inicial
        length = len_circuit(circuit, dist_m)  # Calcula la longitud del circuito
        # Actualiza el circuito mínimo si se encuentra uno más corto
        if length < min_length:
            min_length = length
            min_circuit = circuit

    return min_circuit  # Devuelve el circuito más corto posible
