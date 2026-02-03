import numpy as np
from typing import Optional


class CGA():
    def __init__(self, rows:int, cols:int, maxGeneration:int, 
                 toroidal:bool=False, seed: Optional[int]=None):
        self._random = np.random.default_rng(seed)
        self.rows = rows
        self.cols = cols
        self.toroidal = toroidal
        self.grid:dict = {}
        self.lastGen = maxGeneration
        self.currentGen = 0
        
        # Inicializa população
        for i in range(rows):
            for j in range(cols):
                self.grid[(i, j)] = self.create_individual()
                
    def create_individual(self):
        """Creates a random individual"""
        pass
    
    def get_moore_neighbors(self, pos: tuple[int,int]):
        """Returns neighbors depending of [self.toroidal] value"""
        x = pos[0]
        y = pos[1]
        output = []

        if self.toroidal:
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    if i == 0 and j == 0: continue
                    # Apply Toroidal effect
                    neighPosX = (x + i) % self.rows
                    neighPosY = (y + j) % self.cols
                    output.append((neighPosX,neighPosY))
            return output
        else:
            #Not consider toroidal neighbors
            for i in [-1,0,1]:
                for j in [-1,0,1]:
                    if i==0 and j==0: continue
                    neighPosX, neighPosY = x+i, y+j
                    if (neighPosX < 0 or neighPosY < 0): continue
                    elif (neighPosX > self.rows-1 or neighPosY > self.cols-1): continue
                    output.append((neighPosX,neighPosY))  
            return output
    
    def select(self, neighbors: list[tuple[int,int]]) -> tuple[int,int]:
        return (-1,-1)
    
    def update():
        self.currentGen += 1
        newGrid = {}
        for y in range(self.rows):
            for x in range(self.cols):
                parent1 = self.grid[(x,y)]
                p1Neighbors = self.get_moore_neighbors(pos=(x,y))
                parent2 = self.select(neighbors=p1Neighbors)
                
                
        