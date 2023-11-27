"""
Documentación de las importaciones:

    - typing:
        Este módulo se utiliza para soporte de anotaciones de tipo. La clase 'Tuple'
        se importa para definir tipos de datos compuestos, como tuplas con tipos específicos
        para cada elemento.

    - random: 
        Módulo que implementa generadores de números pseudoaleatorios para diversas
        distribuciones. Se usa comúnmente para la generación de datos aleatorios,
        como pesos en grafos.

    - time:
        Proporciona funciones para trabajar con tiempos, incluyendo funciones para medir
        el tiempo de ejecución, lo cual es útil para el análisis de rendimiento de algoritmos.

    - itertools:
        Ofrece un conjunto de herramientas rápidas y eficientes para crear y manipular iteradores.
        Utilizado para operaciones avanzadas de iteración y manipulación de datos.

    - queue:
        Implementa varias clases de colas, que son útiles para manejar colecciones de 
        elementos en un orden específico. 'PriorityQueue' es una subclase que implementa 
        una cola donde los elementos se ordenan por prioridad.

    - numpy (np):
        Es una biblioteca fundamental para la computación científica en Python.
        Ofrece soporte para arrays y matrices de gran tamaño y herramientas matemáticas
        para trabajar con estos datos.
"""
from typing import Tuple
import random
import time
import itertools
import queue
from queue import PriorityQueue
import numpy as np

# ------------------------- 1A -------------------------


def init_cd(n: int) -> np.ndarray:
    """
    Inicializa un conjunto disjunto para 'n' elementos.
    Cada elemento se representa a sí mismo en la inicialización.

    Args:\n
        n (int): Número de elementos en el conjunto disjunto.

    Returns:\n
        np.ndarray: Un array de NumPy de tamaño 'n' inicializado con -1.
    """
    return np.full(n, -1, dtype=int)


def union(rep_1: int, rep_2: int, p_cd: np.ndarray) -> int:
    """
    Realiza la unión de dos subconjuntos en un conjunto disjunto.

    Args:\n
        rep_1 (int): Representante del primer subconjunto.
        rep_2 (int): Representante del segundo subconjunto.
        p_cd (np.ndarray): Array que representa el conjunto disjunto.

    Returns:\n
        int: El representante del conjunto resultante después de la unión.
    """
    if p_cd[rep_1] > p_cd[rep_2]:
        p_cd[rep_2] = rep_1
        return rep_1
    if p_cd[rep_2] > p_cd[rep_1]:
        p_cd[rep_1] = rep_2
        return rep_2
    p_cd[rep_2] = rep_1
    p_cd[rep_1] -= 1
    return p_cd


def find(ind: int, p_cd: np.ndarray) -> int:
    """
    Encuentra el representante del subconjunto al que pertenece un elemento.

    Args:\n
        ind (int): El índice del elemento.
        p_cd (np.ndarray): Array que9.94/10  representa el conjunto disjunto.

    Returns:\n
        int: El representante del subconjunto al que pertenece el elemento.
    """
    z = ind
    while p_cd[z] >= 0:
        z = p_cd[z]
    while p_cd[ind] >= 0:
        y = p_cd[ind]
        p_cd[ind] = z
        ind = y
    return z


# ------------------------- 2A -------------------------

def create_pq(n: int, l_g: list) -> queue.PriorityQueue:
    """
    Crea una cola de prioridad para los bordes de un grafo.

    Args:\n
        n (int): Número de nodos en el grafo.
        l_g (list): Lista de bordes del grafo, donde cada borde es una tupla (u, v, w).

    Returns:
        queue.PriorityQueue: Una cola de prioridad con todos los bordes insertados.
    """
    pq = PriorityQueue()

    for i in range(0, n):
        for u, v, w in l_g:
            if i == u:
                pq.put((w, u, v))
    return pq


def kruskal(n: int, l_g: list) -> Tuple[int, list]:
    """
    Implementa el algoritmo de Kruskal para encontrar el árbol de expansión mínima de un grafo.

    Args:\n
        n (int): Número de nodos en el grafo.
        l_g (list): Lista de bordes del grafo, donde cada borde es una tupla (u, v, w).

    Returns:\n
        Tuple[int, list]: Peso total del árbol de expansión mínima y lista de bordes en el árbol.
    """
    weight = 0
    tree = init_cd(n)
    l_t = []
    pq = create_pq(n, l_g)
    while not pq.empty():
        (w, u, v) = pq.get()
        if find(u, tree) != find(v, tree):
            l_t.append((u, v))
            weight += w
            union(u, v, tree)
    return weight, l_t


def kruskal_2(n: int, l_g: list) -> Tuple[int, list, float]:
    """
    Versión modificada del algoritmo de Kruskal que también mide el tiempo de ejecución.

    Args:\n
        n (int): Número de nodos en el grafo.
        l_g (list): Lista de bordes del grafo, donde cada borde es una tupla (u, v, w).

    Returns:\n
        Tuple[int, list, float]: Peso total del árbol, lista de bordes en el árbol y
        tiempo total de ejecución.
    """
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
    """
    Genera un grafo completo con pesos aleatorios en los bordes.

    Args:\n
        n_nodes (int): Número de nodos en el grafo.
        max_weight (int, optional): Peso máximo para los bordes del grafo. Por defecto es 50.

    Returns:\n
        Tuple[int, list]: Número de nodos y lista de bordes del grafo generado.
    """
    l_g = []
    for u in range(n_nodes):
        for v in range(u + 1, n_nodes):
            w = random.randint(1, max_weight)
            l_g.append((u, v, w))
    return (n_nodes, l_g)


def time_kruskal(
        n_graphs: int,
        n_nodes_ini: int,
        n_nodes_fin: int,
        step: int) -> list:
    """
    Mide el tiempo promedio de ejecución del algoritmo de Kruskal en una serie de grafos.

    Args:\n
        n_graphs (int): Número de grafos a generar y probar.
        n_nodes_ini (int): Número inicial de nodos en los grafos.
        n_nodes_fin (int): Número final de nodos en los grafos.
        step (int): Incremento en el número de nodos entre grafos sucesivos.

    Returns:\n
        list: Lista de tiempos promedios de ejecución para cada tamaño de grafo.
    """
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
    """
    Mide el tiempo de ejecución del algoritmo de Kruskal y sus operaciones
    internas en una serie de grafos.

    Args:\n
        n_graphs (int): Número de grafos a generar y probar.
        n_nodes_ini (int): Número inicial de nodos en los grafos.
        n_nodes_fin (int): Número final de nodos en los grafos.
        step (int): Incremento en el número de nodos entre grafos sucesivos.

    Returns:\n
        list: Lista de tiempos promedios de ejecución para cada tamaño de grafo.
    """
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
    """
    Genera una matriz de distancias simétrica para un grafo completo.

    Args:\n
        n_nodes (int): Número de nodos en el grafo.
        w_max (int, optional): Peso máximo para los bordes del grafo. Por defecto es 10.

    Returns:\n
        np.ndarray: Matriz de distancias generada.
    """
    m = np.random.randint(1, w_max + 1, (n_nodes, n_nodes))
    m = (m + m.T) // 2
    np.fill_diagonal(m, 0)
    return m


def greedy_tsp(dist_m: np.ndarray, node_ini=0) -> list:
    """
    Implementa un algoritmo greedy para el problema del viajante de comercio (TSP).

    Args:\n
        dist_m (np.ndarray): Matriz de distancias entre nodos.
        node_ini (int, optional): Nodo inicial para el circuito. Por defecto es 0.

    Returns:\n
        list: Un circuito que representa una solución aproximada al TSP.
    """
    num_nodes = dist_m.shape[0]
    circuit = [node_ini]
    while len(circuit) < num_nodes:
        current_node = circuit[-1]
        # sort cities in ascending distance from current
        options = np.argsort(dist_m[current_node])
        # add first city in sorted list not visited yet
        for node in options:
            if node not in circuit:
                circuit.append(node)
                break
    return circuit + [node_ini]


def len_circuit(circuit: list, dist_m: np.ndarray) -> int:
    """
    Calcula la longitud total de un circuito en el problema del viajante de comercio.

    Args:\n
        circuit (list): Circuitos para calcular la longitud.
        dist_m (np.ndarray): Matriz de distancias entre nodos.

    Returns:\n
        int: Longitud total del circuito.
    """
    num_nodes = dist_m.shape[0]
    length = 0
    for j in range(num_nodes):
        length += dist_m[circuit[j]][circuit[j + 1]]
    return length


def repeated_greedy_tsp(dist_m: np.ndarray) -> list:
    """
    Aplica un enfoque greedy para resolver el problema del viajante de comercio (TSP)
    desde cada nodo del grafo, seleccionando el circuito con la menor longitud total.

    Esta función intenta encontrar una solución aproximada al TSP aplicando
    el algoritmo greedy desde cada nodo y eligiendo el mejor circuito encontrado.

    Args:\n
        dist_m (np.ndarray): Matriz de distancias entre todos los pares de nodos.

    Returns:\n
        list: El circuito más corto encontrado, comenzando y terminando en el mismo nodo.
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


def exhaustive_tsp(dist_m: np.ndarray) -> list:
    """
    Resuelve el problema del viajante de comercio (TSP) mediante un enfoque exhaustivo,
    examinando todas las permutaciones posibles de nodos para encontrar el circuito
    con la distancia total más corta.
    Este método es computacionalmente intensivo y solo es práctico para grafos pequeños,
    ya que el número de permutaciones crece factorialmente con el número de nodos.

    Args:\n
        dist_m (np.ndarray): Matriz de distancias entre todos los pares de nodos.
    Returns:\n
        list: El circuito más corto posible, comenzando y terminando en el mismo nodo.
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
