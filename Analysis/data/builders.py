import pandas as pd
import numpy as np
import loaders, tools


################################################################
# builders.py
################################################################
# methods that read data from loaders and adjust to vizualizers
################################################################

def build_directional_hamming_map(df: pd.DataFrame, rows:int, cols: int, toroid:bool=False):
    """Reads the results of a experiment and builds a matrix with directional hamming distances for each generation.

    Output:
    matrix -> a matrix for each generation containing directional hamming distances 
    generations -> a list with each generation in experiment data
    """
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)
    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=dict)

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]
        #build a dict to quick access (faster than filtering dataframe each line)
        shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}
        neighborDistMap = {row["pos"]: {} for _, row in genBots.iterrows()}

        for _, row in genBots.iterrows():
            #for each bot in the generation
            pos = row["pos"]
            matrix[gen, pos[1], pos[0]] = tools.get_directional_hamming_distances(pos, shapeMap, rows, cols, toroid)
         
    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations

def build_fitness_map(df: pd.DataFrame, taskMap:dict, rows:int, cols: int):
    """
    Builds a matrix of (gens, rows, cols) where each cell has the fitness of the robot in its own task"""
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)

    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=float)
    uniqueTasks = set(taskMap.values())
    minmaxDict = {task: {"min": 7777777, "max": -7777777} for task in uniqueTasks}

    #gets the min and max value of fitness in each task
    for task in uniqueTasks:
        all_fits = df['fit'].apply(lambda x: x.get(task, np.nan)).dropna()
        minmaxDict[task] = {
            "min": all_fits.min(),
            "max": all_fits.max()
        }

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]

        #write each bot of this gen
        for _, row in genBots.iterrows():
            x, y = row["pos"]
            pos = f"({x},{y})"
            taskName = taskMap[pos]
            fitValue = row['fit'][taskName]
            # normFit = (fitValue - minmaxDict[taskName]["min"]) / (minmaxDict[taskName]["max"] - minmaxDict[taskName]["min"])
            matrix[gen, y, x] = fitValue

    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations, minmaxDict

if __name__=="__main__":
    matrix, gen = build_directional_hamming_map(experimentFolder="log/v1/quadrantv1_seed7_CGA_04302108", toroid=False)