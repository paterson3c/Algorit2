import numpy as np

def init_cd(n: int)-> np.ndarray:
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

    if p_cd[ind] < 0:
        return ind
    else:
        find(p_cd[ind], p_cd)

if (__name__ == "__main__"):
    print("Ernes es un putero, los bots le hablan al dm")