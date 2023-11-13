import numpy as np
import p131A as cd
import queue
from queue import PriorityQueue
from typing import Tuple
import time

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
    tree = cd.init_cd(n)
    l_t = []


    pq = create_pq(n, l_g)

    while not pq.empty():
        (w, u, v) = pq.get()

        if(cd.find(u, tree) != cd.find(v, tree)):
            l_t.append((u, v))
            weight += w
            cd.union(u, v, tree)
        
    return weight, l_t

def kruskal_2(n: int, l_g: list)-> Tuple[int, list, float]:
    weight = 0
    tree = cd.init_cd(n)
    l_t = []
    total = 0


    pq = create_pq(n, l_g)

    while not pq.empty():
        (w, u, v) = pq.get()

        start = time.time()
        f1 = cd.find(u, tree)
        end = time.time()
        total += end - start

        start = time.time()
        f2 = cd.find(v, tree)
        end = time.time()
        total += end - start

        if( f1 != f2):
            l_t.append((u, v))
            weight += w
            start = time.time()
            cd.union(u, v, tree)
            end = time.time()
            total += end - start
        
    return weight, l_t, total


if __name__ == "__main__":
    n = 4
    l_g = [(0, 1, 5), (1, 2 ,4), (2, 3, 1), (3, 0, 2), (0, 2, 1), (1, 3, 5)]

    pq = create_pq(n, l_g)

    print(pq.qsize())
    print_pq_contents(pq)
    print(pq.qsize())