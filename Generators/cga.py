import numpy as np
import Search
from typing import Optional
from robot.basicrobot import SinRobot as Robot, get_random, get_fromfile


class CGA():
    def __init__(self, rows:int, cols:int, maxGeneration:int, toroidal:bool=False, mutationChance:float = 0.05, seed: Optional[int]=None):
        self._random = np.random.default_rng(seed)
        self.rows = rows
        self.cols = cols
        self.toroidal = toroidal
        self.grid:dict[tuple[int,int],'Robot'] = {}
        self.lastGen = maxGeneration
        self.currentGen:int = 0
        self.mutationChance = mutationChance
        
        # Start population
        for i in range(rows):
            for j in range(cols):
                self.grid[(i, j)] = self.get_random_cromossome()
                
    def get_random_cromossome(self) -> 'Robot':
        """Creates a random individual"""
        return Robot()     
    
    def get_moore_neighbors(self, pos: tuple[int,int]) -> list[tuple[int,int]]:
        """Returns moore neighbors depending of [self.toroidal] value"""
        x = pos[0]
        y = pos[1]
        output:list[tuple[int,int]] = []
            #Consider toroidal neighbors
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
    
    def save_grid(self, address:str, grid2Save:dict[tuple[int,int],'Robot']) -> None:
        pass
    
    def evaluate(self, chromossome:'Robot') -> float:
        #Evaluates cromossome, change fitness inside itself, return calculated fitness
        return 0.75
        
    def select(self, neighbors: list[tuple[int,int]]) -> tuple[int,int]:
        return (-1,-1)
    
    def update(self):        
        self.currentGen += 1
        newGrid = {} 
        for y in range(self.rows):
            for x in range(self.cols):
                parent1 = self.grid[(x,y)]
                p1Neighbors = self.get_moore_neighbors(pos=(x,y))
                parent2 = self.grid[self.select(neighbors=p1Neighbors)]
                child = parent1.crossover(mate=parent2)
                if self._random.random() <= self.mutationChance:
                    child = child.mutate()
                childFit = self.evaluate(chromossome=child)
                if childFit >= parent1.fit:
                    newGrid[(x,y)] = child
                else:
                    newGrid[(x,y)] = parent1
        
        self.grid = newGrid
                    
