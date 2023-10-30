import numpy as np
import p131A as cd
import queue
from queue import PriorityQueue
from typing import Tuple

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
    l_t = []

    pq = create_pq(n, l_g)

    while not pq.empty():
        item = pq.get()
                

    return



if __name__ == "__main__":
    n = 4
    l_g = [(0, 1, 5), (1, 2 ,4), (2, 3, 1), (3, 0, 2), (0, 2, 1), (1, 3, 5)]

    pq = create_pq(n, l_g)

    print(pq.qsize())
    print_pq_contents(pq)
    print(pq.qsize())