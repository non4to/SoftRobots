import numpy as np, time
import Search as Search
from typing import Optional
from multiprocessing import Pool
from robot.basicrobot import SinRobot as Robot, get_random, get_fromfile

class CGA():
    def __init__(self, numprocs, robotModule, worldModule, sim_step:int, size:int, maxGeneration:int, toroidal:bool=False, mutationChance:float = 0.05, rng: Optional[np.random.Generator]=None):
        self._random = rng if rng is not None else np.random.default_rng()
        self._robot = robotModule
        self._world = worldModule
        self._sim_step = sim_step
        self._numprocs = numprocs
        self.rows = size
        self.cols = size
        self.toroidal = toroidal
        self.grid:dict[tuple[int,int],'Robot'] = {}
        self.lastGen = maxGeneration
        self.mutationChance = mutationChance
                        
        self.currentGen:int = 0
        self.meanTime = []

        #Best Dict
        self._bestDict = {"pos":(-1,-1), "fit":-1, "robot":None}    
    
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
    
    def reset(self) -> None:
        evalpars = []
        botsList = []

        #random bots as start population
        for y in range(self.rows):
            for x in range(self.cols):
                robot = self._robot.get_random(rng=self._random)
                botsList.append(((x,y), robot))
                evalpars.append((robot, self._world, self._sim_step))

        #Evaluate bots
        with Pool(self._numprocs) as p:
            scores = p.starmap(Search.evaluate, evalpars)
        
        for s in scores:
            self.meanTime.append(s[1])

        #fill grid
        for i, (pos, bot) in enumerate(botsList):
            bot.fit = scores[i][0]
            self.check_best(bot, pos)
            self.grid[pos] = bot 

    def save_grid(self, address:str) -> None:
        pass
        
    def select(self, neighbors: list[tuple[int,int]]) -> tuple[int,int]:
        bots = [self.grid[pos] for pos in neighbors]
        bots = sorted(bots, key=lambda bot: bot.fit, reverse=True)
        a=1

        return neighbors[0]
    
    def check_best(self, newrobot, pos):
        ####
        if newrobot.fit > self._bestDict["fit"]:
            self._bestDict["fit"] = newrobot.fit
            self._bestDict["pos"] = pos
            self._bestDict["robot"] = newrobot
            print(f"New best found in Generation {self.currentGen} at {self._bestDict['pos']}: {self._bestDict['fit']}")
        ####

    def update(self):        
        self.currentGen += 1
        print(f"Gen: {self.currentGen}")
        childrenList = []
        evalpars = []

        #generate children
        for y in range(self.rows):
            for x in range(self.cols):
                parent1 = self.grid[(x,y)]
                p1Neighbors = self.get_moore_neighbors(pos=(x,y))
                parent2 = self.grid[self.select(neighbors=p1Neighbors)]
                child = parent1.crossover(mate=parent2)
                if self._random.random() <= self.mutationChance:
                    child = child.mutate()
                childrenList.append(((x,y), child))
                evalpars.append((child, self._world, self._sim_step))
                # child.fit,_ = self.evaluate(robot=child)
                # self.check_best(child, (x,y))
                # if child.fit >= parent1.fit:
                #     
                # else:
                #     children[(x,y)] = parent1

        #evaluate children
        with Pool(self._numprocs) as p:
            scores = p.starmap(Search.evaluate, evalpars)

        for s in scores:
            self.meanTime.append(s[1])

        #fill new grid
        newGrid = {}
        for i, (pos, child) in enumerate(childrenList):
            child.fit = scores[i][0]
            self.check_best(child, pos)
            parent = self.grid[pos]
            newGrid[pos] = child if child.fit >= parent.fit else parent

        self.grid = newGrid
                    
