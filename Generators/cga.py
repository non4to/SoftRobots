import numpy as np, time, json, time, os
import Search as Search
from typing import Optional
from multiprocessing import Pool
from robot.basicrobot import SinRobot as Robot, get_random, get_fromfile

class CGA():
    def __init__(self, logdir, prefix, save_interval, numprocs, robotModule, worldModules, sim_step:int, gridWorlds:list[list[int]], maxGeneration:int, toroidal:bool=False, mutationChance:float = 0.05, rng: Optional[np.random.Generator]=None):
        self._random = rng if rng is not None else np.random.default_rng()
        self._robot = robotModule
        self._worlds = worldModules
        self._gridWorlds = gridWorlds
        self.rows = len(self._gridWorlds) #assumes rectangle, always.
        self.cols = len(self._gridWorlds[0])
        self._sim_step = sim_step
        self._numprocs = numprocs
        self._prefix = prefix
        self._logdir = logdir
        self._save_interval = save_interval
        self.toroidal = toroidal
        self.grid:dict[tuple[int,int],'Robot'] = {}
        self.lastGen = maxGeneration
        self.mutationChance = mutationChance
                        
        self.currentGen:int = 0
        self.robotCounter:int = 0
        self.meanTime = []

        #Best Dict
        self._bestDict = {"pos":(-1,-1), "fit":-1, "robot":None}  
        self.buildTaskMap()

    def buildTaskMap(self):
        self._taskMap = {}
        for y in range(self.rows):
            for x in range(self.cols):
                index = self._gridWorlds[y][x]
                self._taskMap[(x,y)] = self._worlds[index]

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

        # os.makedirs(self._prefix, exist_ok=True)

        #random bots as start population
        for y in range(self.rows):
            for x in range(self.cols):
                robot = self._robot.get_random(rng=self._random)
                robot.id = self.robotCounter
                self.robotCounter += 1
                botsList.append(((x,y), robot))
                evalpars.append((robot, self._taskMap[(x,y)], self._sim_step))

        #Evaluate bots
        with Pool(self._numprocs) as p:
            scores = p.starmap(Search.evaluate, evalpars)
        
        for s in scores:
            self.meanTime.append(s[1])

        #fill grid
        gridSnapShot = {}
        for i, (pos, bot) in enumerate(botsList):
            bot.fit = scores[i][0]
            self.check_best(bot, pos, "None")
            self.grid[pos] = bot 
            bot.pos = pos

            key = f"({pos[0]},{pos[1]})"
            gridSnapShot[key] = {"id": bot.id,
                                "class": bot.__class__.__name__,
                                "shape": bot.shape.tolist(),
                                "parent": "None"}

        with open(f"{self._prefix}{os.sep}grid_startPop.json", "w") as out_f:
            json.dump(gridSnapShot, out_f, separators=(',', ':'))

        jsonTaskMap = {}
        for (x,y) in self._taskMap:
            jsonTaskMap[f"({x},{y})"] = type(self._taskMap[(x,y)]).__module__

        with open(f"{self._prefix}{os.sep}grid_taskMap.json", "w") as out_f2:
            json.dump(jsonTaskMap, out_f2, separators=(',', ':'))

    def check_best(self, newrobot, pos, parentPos):
        ####
        if newrobot.fit > self._bestDict["fit"]:
            self._bestDict["fit"] = newrobot.fit
            self._bestDict["pos"] = pos
            self._bestDict["robot"] = newrobot
            print(f"New best found in Generation {self.currentGen} at {self._bestDict['pos']}: {self._bestDict['fit']}")
            
            newrobot.save_json(f"{self._prefix}{os.sep}robot_{pos[0]}_{pos[1]}_gen{self.currentGen}.json",
                            {"parent":parentPos})
        ####

    def select(self, neighbors: list[tuple[int,int]]) -> tuple[int,int]:
        # bots = [self.grid[pos] for pos in neighbors]
        # sortedBots = sorted(bots, key=lambda bot: bot.fit, reverse=True)
        sortedNeighbors = sorted(neighbors, key=lambda pos: self.grid[pos].fit, reverse=True)
        
        n = len(neighbors)
        weights = [1 / (2**i) for i in range(n)]
        weights[-1] = weights[-2]/2
        total = sum(weights)
        weights = [w/total for w in weights]
        
        picked = self._random.choice(len(neighbors), p=weights)

        return sortedNeighbors[picked]
    
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
                parent2pos = self.select(neighbors=p1Neighbors)
                parent2 = self.grid[parent2pos]
                child = parent1.crossover(mate=parent2)
                if self._random.random() <= self.mutationChance:
                    child = child.mutate()

                child.id = self.robotCounter
                self.robotCounter += 1
                childrenList.append(((x,y), child, parent2pos))
                evalpars.append((child, self._world, self._sim_step))

        #evaluate children
        with Pool(self._numprocs) as p:
            scores = p.starmap(Search.evaluate, evalpars)
        for s in scores:
            self.meanTime.append(s[1])

        #fill new grid
        newGrid = {}
        gridSnapShot = {}
        for i, (pos, child, parentPos) in enumerate(childrenList):
            child.fit = scores[i][0]
            self.check_best(child, pos, parentPos)
            parent = self.grid[pos]
            newGrid[pos] = child if child.fit >= parent.fit else parent

            #save a bot
            # child.save_json(f"{self._prefix}{os.sep}robot_{pos[0]}_{pos[1]}_gen{self.currentGen}.json",
            #                 {"parent":parentPos})

            #fill snapshotgrid if is to be saved
            if self.currentGen % self._save_interval == 0:
                key = f"({pos[0]},{pos[1]})"
                gridSnapShot[key] = {"id": child.id,
                                    "class": child.__class__.__name__,
                                    "fit": child.fit,
                                    "parent": parentPos,
                                    "shape": child.shape.tolist()}
        #save whole grid if is to be saved
        if self.currentGen % self._save_interval == 0:
            with open(f"{self._prefix}{os.sep}grid_gen{self.currentGen}.json", "w") as out_f:
                json.dump(gridSnapShot, out_f, separators=(',', ':'))

        #update current grid for next gen
        self.grid = newGrid
                    
