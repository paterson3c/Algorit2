import numpy as np
import p131A as cd
import p132A as kruskal
from queue import PriorityQueue
from typing import Tuple
import random
import time

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
            kruskal.kruskal(n_nodes, l_g)
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
            tw, tree, time = kruskal.kruskal_2(n_nodes, l_g)
            total += time

        l_times.append((n, (total / n_graphs)))
        n_nodes += step

    return l_times


if __name__ == "__main__":
    n_graphs = 10
    n_nodes_ini = 3
    n_nodes_fin = 10
    step = 1

    l_times = time_kruskal(n_graphs, n_nodes_ini, n_nodes_fin, step)
    l_times2 = time_kruskal_2(n_graphs, n_nodes_ini, n_nodes_fin, step)

    print(l_times)
    print(l_times2)
