import pytest, sys, numpy as np
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from Generators.cga import CGA 

def test_neighbors():
    cga1 = CGA(4, 4, toroidal=False)
    assert(set(cga1.get_moore_neighbors((1,1)))=={(0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)})
    assert(set(cga1.get_moore_neighbors((2,1)))=={(1,0),(2,0),(3,0),(1,1),(3,1),(1,2),(2,2),(3,2)})
    assert(set(cga1.get_moore_neighbors((0,0)))=={(1,0),(1,1),(0,1)})
    assert(set(cga1.get_moore_neighbors((3,3)))=={(3,2),(2,2),(2,3)})
        
    cga2 = CGA(4, 4, toroidal=True)
    assert(set(cga2.get_moore_neighbors((1,1)))=={(0,0),(1,0),(2,0),(0,1),(2,1),(0,2),(1,2),(2,2)})
    assert(set(cga2.get_moore_neighbors((2,1)))=={(1,0),(2,0),(3,0),(1,1),(3,1),(1,2),(2,2),(3,2)})
    assert(set(cga2.get_moore_neighbors((0,0)))=={(1,0),(1,1),(0,1),(0,3),(1,3),(3,0),(3,1),(3,3)})
    assert(set(cga2.get_moore_neighbors((3,3)))=={(3,2),(2,2),(2,3),(0,0),(0,3),(0,2),(3,0),(2,0)})



# def test_neighbors_corner_non_toroidal():
#     cga = CGA(5, 5, toroidal=False)
#     neighbors = cga.get_moore_neighbors((0, 0))
#     assert len(neighbors) == 3
#     assert set(neighbors) == {(0,1), (1,0), (1,1)}

# def test_neighbors_corner_toroidal():
#     cga = CGA(5, 5, toroidal=True)
#     neighbors = cga.get_moore_neighbors((0, 0))
#     assert len(neighbors) == 8
#     assert (4, 4) in neighbors
#     assert (0, 1) in neighbors
#     assert (1, 0) in neighbors
