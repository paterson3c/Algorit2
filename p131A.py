import numpy as np

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

if (__name__ == "__main__"):
    print("This is a module, not a program")