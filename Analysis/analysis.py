import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seaborn as sns
import pandas as pd, numpy as np
import json, imageio, importlib, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pygifsicle import optimize

EXPERIMENT_NAMES = ['baseline-walkerv0','quadrant-v0_childFirst','quadrantv0',
 'baseline-BridgeWalker_v0']

SEEDS = ['8','16807','49','4096','32768','64','7','343','2401','512']

# MIN_MAX_FOUND = {'fit_world.Walker_v0': [-66.30151741956223, 115.50124506255865], 
#                  'fit_world.BridgeWalker_v0': [-78.46431125410749, 93.85425490887846]}

HATCHING_PATTERNS = {
    0: '',     
    1: '/////',      
    2: '|||',    
    3: '***',    
    4: '---',
}

def load_parquet_log(archivePath:str):
    """Returns (escalated fitness with min max found for each task)
    df -> read dataframe
    fitNames -> names of columns that have task fitness
    minmaxValues -> minimum and maximum found for each task found in df"""
    df = pd.read_parquet(archivePath)
    df = df.loc[df['experiment'] != 'quadrant-v0_childFirst'].copy()

    fitNames = []
    minmaxValues = {}

    for colName in df.columns:
        if "fit_" in colName:
            fitNames.append(colName)
            minmaxValues[colName] = []
            maxFit = df[colName][df[colName].idxmax()]
            minFit = df[colName][df[colName].idxmin()]
            minmaxValues[colName] = [minFit,maxFit]  
            df[colName] = (df[colName] - minFit) / (maxFit-minFit)
    
    return df, fitNames, minmaxValues

def load_log(logdir: str):
    """logdir: Log's root folder
    Returns: [Pandas Dataframe of Robots Logs, Dict of TaskMap, Tuple with grid size]
        """
    #loads log to pandas
    bots_path = os.path.join(logdir, "robots_log.jsonl")
    with open(bots_path, "r") as f:
        df = pd.read_json(f, lines=True)
    df["pos"] = df["pos"].apply(tuple)
    
    #loads taskMap to dict
    taskmap_path = os.path.join(logdir, "grid_taskMap.json")
    with open(taskmap_path, "r") as f:
        taskMap = json.load(f)
        
    #get grid size
    positions = [
        tuple(int(v) for v in key.strip("()").split(","))
        for key in taskMap.keys()
    ]
    cols = max(x for x, y in positions) + 1
    rows = max(y for x, y in positions) + 1

    #adjust generations: there is a group in the end with gen = 99999. these robots were evaluated on tasks they didnt perform during the algorithm. this is just to identify them
    #this code puts their fitness in the corresponding robot in the final generation
    realGens = df[df["gen"] != 99999]["gen"].unique() #only the real gens from the exp, ignoring the 99999 one used to evaluate bots that needed evaluation
    lastRealGen = max(realGens)                       #check the real last gen after taking 99999 off
    extraEvals = df[df["gen"] == 99999][["id", "fit", "pos"]].set_index("id") #creates a new df with only the lines where gen is 99999. Gets only some columns

    for botId, extraFit in extraEvals["fit"].items():
        mask = (df["id"] == botId) & (df["gen"] == lastRealGen) #mask is the bot with the same id and same lastgen
        if mask.any(): #if this bot is found
            df.loc[mask, "fit"] = df.loc[mask, "fit"].apply(lambda oldFitDict: {**oldFitDict, **extraFit}) #for the bot where the mask is true,  it puts the keys of extraFit in oldFitDict
    
    df = df[df["gen"] != 99999] #erases the lines with the extra gen because we got this data
    return df, taskMap, (rows, cols)

def get_directional_hamming_distances(referencePos:tuple[int,int], shapeMap:dict, rows:int, columns:int, toroidal:bool=False) -> dict:
    """Returns a dict where the keys are the position of neighbors of [referencePos] and the content is the hamming distance between them"""
    outputDict = {}
    neighbors = get_moore_neighbors(referencePos, rows, columns, toroidal)
    for neighbor in neighbors:
        if neighbor not in outputDict: outputDict[neighbor] = 0
        outputDict[neighbor] = hamming_distance(shape1=shapeMap[referencePos], shape2=shapeMap[neighbor])
    return outputDict

def build_actuator_maps(df: pd.DataFrame, rows:int, cols:int):
    """
    Builds a matriz of (gens, rows, cols) with the numbers of actuators;
    Returns matrix and max/min values of actuators
    """
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)

    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=float)

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]

        #write each bot of this gen
        for _, row in genBots.iterrows():
            x, y = row["pos"]
            matrix[gen, y, x] = row["actuators"]

    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")

    minValue = matrix.min()
    maxValue = matrix.max()
    
    return matrix, generations, maxValue, minValue

def build_hamming_distance_map(df: pd.DataFrame, rows:int, cols: int, toroid:bool=False):
    """
    Builds a matrix of (gens, rows, cols) where each cell has the avg hamming distance value between a cell and its neighbors.
    0.0 identical to all neighbors
    1.0 completely different from all neighbors
    """
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)

    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=float)

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]
        #build a dict to quick access (faster than filtering dataframe each line)
        shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}

        for _, row in genBots.iterrows():
            #for each bot in the generation
            pos = row["pos"]
            thisShape = row["shape"]
            neighbors = get_moore_neighbors(pos=row["pos"], rows=rows, cols=cols, toroidal=toroid)
            distances2neigh = [] 
            #for this bot's neighbors
            for neigh in neighbors: #neighbors is a list of positions
                if neigh in shapeMap:
                    dist = hamming_distance(thisShape, shapeMap[neigh])
                    distances2neigh.append(dist)

            if distances2neigh:
                x, y = pos
                matrix[gen, y, x] = np.mean(distances2neigh)

    #check for missing values in the matrix
    missing = np.sum(matrix == -1)
    if missing > 0:
        print(f"Missing values: {missing}")    

    return matrix, generations, matrix.max(), matrix.min()

#TODO: Read this function and start creating the print one
def build_global_hamming_distance_map(df: pd.DataFrame, rows:int, cols: int):
    """
    Builds a matrix of (gens, rows, cols) where each cell has the avg hamming distance value between all cells in the map.
    0.0 identical 
    1.0 completely different
    """
    #grabs all present generations
    generations = sorted(df["gen"].unique())
    n_gens = len(generations)

    #starts matrix with -1 to indicate stuff that was not found in dataframe
    matrix = np.full((n_gens, rows, cols), fill_value=-1, dtype=float)

    #iterates throught generations and builds matrix
    for gen in generations:
        #get robots of this gen
        genBots = df[df["gen"]==gen]
        #build a dict to quick access (faster than filtering dataframe each line)
        shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}

        for _, row in genBots.iterrows():
            #for each bot in the generation
            pos = row["pos"]
            thisShape = row["shape"]
            distances2all = [] 
            #for all other bots
            for _, otherRow in genBots.iterrows():
                dist = hamming_distance(thisShape, shapeMap[otherRow["pos"]])
                distances2all.append(dist)

            if distances2all:
                x, y = pos
                matrix[gen, y, x] = np.mean(distances2all)

    #check for missing values in the matrix
    missing = np.sum(matrix == -1)
    if missing > 0:
        print(f"Missing values: {missing}")    

    return matrix, generations, matrix.max(), matrix.min()

def build_fitness_data(df: pd.DataFrame, taskMap:dict) -> dict:
    """
    Reads the robots log dataframe and builds one dataset per task.
    Each dataset contains the mean fitness and std deviation per generation.

    Returns a list of dicts with format:
        {
            "label": str,        # task name (short)
            "x": list[int],      # generation indices
            "y": list[float],    # mean fitness per generation (across seed)
            "std": list[float],  # std deviation across seeds (or within itself if only seed)
        }
    """
    # lista com todas as gerações
    # para cada geração, passar por cada celular e pegar o valor de fitness da task daquela célula e colocar numa lista.
    # tirar média e desvio padrão
    generations = sorted(df["gen"].unique())
    seeds = sorted(df["seed"].unique())
    uniqueTasks = list(set(taskMap.values()))
    minmaxDict = {task: {"min": 7777777, "max": -7777777} for task in uniqueTasks}
    output = {}
    for task in uniqueTasks:
        if task not in output: output[task] = {}
        output[task]["label"] = task.split(".")[1]
        output[task]["x"] = []
        output[task]["y"] = []
        output[task]["std"] = []    

    #gets the min and max value of fitness in each task
    for task in uniqueTasks:
        all_fits = df['fit'].apply(lambda x: x.get(task, np.nan)).dropna()
        minmaxDict[task] = {
            "min": all_fits.min(),
            "max": all_fits.max()
        }

    for genIdx, gen in enumerate(generations):
        genBots = df[df["gen"]==gen]
        valueList = {}

        for _, row in genBots.iterrows():
            x, y = row["pos"]
            pos = f"({x},{y})"
            taskName = taskMap[pos]
            if taskName not in valueList: valueList[taskName] = []
            fitValue = row['fit'][taskName]
            normFit = (fitValue - minmaxDict[taskName]["min"]) / (minmaxDict[taskName]["max"] - minmaxDict[taskName]["min"])
            valueList[taskName].append(normFit)

        for task in output:
            avgValue = np.mean(valueList[task])
            stdValue = np.std(valueList[task])

            output[task]["x"].append(gen)
            output[task]["y"].append(avgValue)
            output[task]["std"].append(stdValue)
    return output    

def build_hamming_data (df: pd.DataFrame, taskMap: dict) -> dict:
    """
    Reads the robots log dataframe and builds a dataset of hamming distance per gen
    Each dataset contains the mean fitness and std deviation per generation.

    Returns a list of dicts with format:
        {
            "label": str,        # 
            "x": list[int],      # generation indices
            "y": list[float],    # mean fitness per generation
            "std": list[float],  # std deviation per generation
        }
    """
    generations = sorted(df["gen"].unique())
    uniqueTasks = list(set(taskMap.values()))

    output = {"global": {"label": "global", "x": [], "y": [], "std": []}}
    for task in uniqueTasks:
        if task not in output: output[task] = {}
        output[task]["label"] = task.split(".")[1]
        output[task]["x"] = []
        output[task]["y"] = []
        output[task]["std"] = []    

    for genIdx, gen in enumerate(generations):
        genBots = df[df["gen"]==gen]
        shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}
        taskGroups = {task: [] for task in uniqueTasks}

        #separate bots by task
        for _, row in genBots.iterrows():
            x, y   = row["pos"]
            posKey = f"({x},{y})"
            taskGroups[taskMap[posKey]].append(row["pos"])

        #get hamming by task
        for task in uniqueTasks:
            positions = taskGroups[task]
            distances = []

            for i in range(len(positions)):
                for j in range(i+1, len(positions)): #compare inside same task, where i!=j, and only once (ij=ji)
                   bot1 = shapeMap[positions[i]]
                   bot2 = shapeMap[positions[j]]
                   dist = hamming_distance(bot1,bot2)
                   distances.append(dist)

            avgDist = np.mean(distances)
            stdDist = np.std(distances)
            output[task]["x"].append(gen)
            output[task]["y"].append(avgDist)
            output[task]["std"].append(stdDist)

        #global hamming
        allPositions = list(shapeMap.keys())
        globalDistances = []
        for i in range(len(allPositions)):
            for j in range(i + 1, len(allPositions)):
                bot1 = shapeMap[allPositions[i]]
                bot2 = shapeMap[allPositions[j]]
                dist = hamming_distance(bot1,bot2)
                distances.append(dist)

        avgDist = np.mean(distances)
        stdDist = np.std(distances)
        output["global"]["x"].append(gen)
        output["global"]["y"].append(avgDist)
        output["global"]["std"].append(stdDist)

    return output

def build_fit_scatter_data (df: pd.DataFrame, taskMap: dict) -> dict:
    """
    Uses the final generation data (which includes cross-task evaluations
    merged by load_log) to analyze specialist vs generalist robots.

    Returns:
        "robots": [
                {
                    "pos":       (x, y),
                    "localTask": str,        # task assigned to this cell
                    "taskNameA":      float,      # normalized fitness on taskA
                    "taskNameB":      float,      # normalized fitness on taskB
                    "delta":     float,      # |fitA - fitB| specialization score
                },
                ...
            ]
    """
    uniqueTasks = sorted(set(taskMap.values()))
    lastGen     = df["gen"].max()
    lastBots    = df[df["gen"] == lastGen]

    minmaxDict = {}
    for task in uniqueTasks:
        all_fits = df["fit"].apply(lambda x: x.get(task, np.nan)).dropna()
        minmaxDict[task] = {"min": all_fits.min(), "max": all_fits.max()}

    bots = []
    for _, row in lastBots.iterrows():
        x, y = row["pos"]
        pos = f"({x},{y})"
        localTask = taskMap[pos]
        bot = {
            "pos": row["pos"],
            "localTask": localTask,
        }

        fitValue = row['fit']        
        for taskName in fitValue.keys():
            bot[taskName] = -1
            normFit = (fitValue[taskName] - minmaxDict[taskName]["min"]) / (minmaxDict[taskName]["max"] - minmaxDict[taskName]["min"])
            bot[taskName] = normFit
        
        bot["delta"] = abs(bot[uniqueTasks[0]] - bot[uniqueTasks[1]])
        bots.append(bot)
    return {"tasks": uniqueTasks, "bots":bots}

def build_directional_hamming_map(df: pd.DataFrame, rows:int, cols: int, toroid:bool=False):
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
            matrix[gen, pos[1], pos[0]] = get_directional_hamming_distances(pos, shapeMap, rows, cols, toroid)
         
    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations

def build_fitness_map(df: pd.DataFrame, taskMap:dict, rows:int, cols: int):
    """
    Builds a matrix of (gens, rows, cols) where each cell has the fitness of the robot in its own task
    Each value is normalized according to the max and min of each task."""
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
            normFit = (fitValue - minmaxDict[taskName]["min"]) / (minmaxDict[taskName]["max"] - minmaxDict[taskName]["min"])
            matrix[gen, y, x] = normFit

    missing = np.sum(matrix == -1)
    if missing > 0: print("Missing values in matrix!")
    return matrix, generations, minmaxDict

def build_task_overlay(taskMap:dict, rows:int, cols:int):
    """Returns a matrix assigning an int for each task in the taskMap
    """
    taskNames = sorted(set(taskMap.values())) #sort alphabetically to guarantee same order in different executions
    overlayMatrix = np.zeros((rows, cols), dtype=int)
        
    for key, task in taskMap.items():
        x, y = key.strip("()").split(",")
        overlayMatrix[int(y),int(x)] = taskNames.index(task)
        
    return overlayMatrix, taskNames    

def render_generation_map(
    targetMatrix: np.ndarray, #matrix with relevant data; ex: number of actuators for each cell in the grid
    taskMatrix: np.ndarray,   #matrix that indicates which task is in which cell
    taskNames: list,          #task names in the correct order - indexes here indicate tasks in taskMatrix
    minValue: float,
    maxValue: float,          #min and max value of targetMatrix, used for heatmap.
    gen: int,                 #generation of this map
    taskColors: list[str],    #colors of each task in taskNames (for rectangle) 
    legendText: str,          #text that appears beside legends
    figSize: tuple = (10,10) ) -> np.ndarray:
    
    plt.close("all")
    rows, cols = targetMatrix.shape
    fig, ax = plt.subplots(figsize = figSize)

    #set heatmap with max and min values
    im = ax.imshow(
        targetMatrix,
        cmap='seismic',
        vmin=minValue,
        vmax=maxValue,
        aspect="equal"
    )
    
    #print borders for different tasks
    for y in range(rows):
        for x in range(cols):


            taskIndex = taskMatrix[y, x]
            color = taskColors[taskIndex]
            
            rect = patches.Rectangle(
                (x - 0.5, y - 0.5), 
                1, 1,                  
                linewidth=3,
                edgecolor=color,
                facecolor="none")
            ax.add_patch(rect)

    #value inside cell
    for y in range(rows):
        for x in range(cols):
            ax.text(
                x, y,
                f"{round(targetMatrix[y, x],2)}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="black")
            
    #color bar and title
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(legendText, fontsize=10)

    #task subtitles
    legendElements = [
        patches.Patch(edgecolor=taskColors[i], facecolor="none",
                      linewidth=3, label=taskNames[i].split(".")[-1])
        for i in range(len(taskNames))]
    ax.legend(handles=legendElements, loc="upper left",
              bbox_to_anchor=(1.15, 1.0), fontsize=9)

    ax.set_title(f"Generation {gen}", fontsize=13)
    ax.axis("off")

    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())  # shape (h, w, 4) — RGBA
    frame = frame[:, :, :3]                        # descarta canal alpha → RGB
    plt.close(fig)
    return frame

def render_hamming_direction_map(
    fitnessMatrix: np.ndarray, #output of build_fitness_map
    fitMinValue: float,
    fitMaxValue: float,        #min and max value of targetMatrix, used for heatmap.
    directionalMatrix: np.ndarray, #output of build_directional_hamming_map
    taskMatrix: np.ndarray,   #matrix that indicates which task is in which cell
    taskNames: list,          #task names in the correct order - indexes here indicate tasks in taskMatrix
    gen: int,                 #generation of this map
    taskColors: list[str],    #colors of each task in taskNames (for rectangle) 
    legendText: str,          #text that appears beside legends
    figSize: tuple = (10,10) ) -> np.ndarray:

    plt.close("all")
    rows, cols = directionalMatrix.shape
    fig, ax = plt.subplots(figsize = figSize)

    #set heatmap with max and min values
    im = ax.imshow(
        fitnessMatrix,
        cmap='bwr',
        vmin=fitMinValue,
        vmax=fitMaxValue,
        aspect="equal"
    )

    #set limits
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect('equal')

    #draw hamming lines
    for y in range(rows):
        for x in range(cols):
            neighDict = directionalMatrix[y, x]
            # if neighDict is None or isinstance(neighDict, float): continue
            #to each cell neighbor
            for (neighX, neighY), hamming in neighDict.items():
                # if hamming is None or hamming < 0: continue
                if hamming < 0.15:
                    alpha = 1
                    linewidth = 7
                elif hamming < 0.5:
                    alpha = 0.4
                    linewidth = 5
                else:
                    alpha = 0
                    linewidth = 0
                                
                x_end = x + (neighX - x) * 0.4
                y_end = y + (neighY - y) * 0.4

                ax.plot([x, x_end], [y, y_end],
                        color='black', alpha=alpha, linewidth=linewidth,
                        solid_capstyle='round')
    
    #draw task borders
    for y in range(rows):
        for x in range(cols):
            taskIndex = taskMatrix[y, x]
            pattern = HATCHING_PATTERNS.get(taskIndex, '')
            rect = patches.Rectangle(
            (x - 0.5, y - 0.5), 
            0.99, 0.99,                  
            linewidth=1,
            edgecolor='black',
            facecolor='none',
            hatch=pattern,
            alpha=0.4)
            ax.add_patch(rect)

    #color bar and title
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(legendText, fontsize=10)

    #task subtitles
    legendElements = [
        patches.Patch(hatch=HATCHING_PATTERNS.get(i, ''),
        facecolor="white",
        edgecolor="black", linewidth=1,
        label=taskNames[i].split(".")[-1])
        for i in range(len(taskNames))]
    ax.legend(handles=legendElements, loc="upper left",
              bbox_to_anchor=(1.25, 1.0), fontsize=9)

    #render
    ax.set_title(f"Generation {gen}", fontsize=13)
    ax.axis("off")
    
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = frame[:, :, :3]  # RGB apenas
    plt.close(fig)
    return frame

def print_actuators_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskMap:dict, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the amount of actuators in each robot.
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(characterize_bot).apply(pd.Series)
    # df = pd.concat([df, newCols], axis=1)

    #builds actuators maps
    matrix, generations, maxValue, minValue = build_actuator_maps(df, rows, cols)
    overlayMatrix, taskNames = build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []
    
    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(generations):
        isLastGen = (g_idx == len(generations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_generation_map(
                matrix[g_idx], overlayMatrix, taskNames,
                minValue, maxValue, gen, taskColors, "Number of Actuators",
                figSize=(8,8))
            frames.append(frame)
        
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(generations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "heatmap_actuators.gif")
    imageio.mimsave(output_path, frames, duration=frameDuration)  # frameDurationms por frame
    print(f"GIF salved in: {output_path}")

def print_hammming_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskMap:dict, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the hammming distance of a cell to its neighbors
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(characterize_bot).apply(pd.Series)
    # df = pd.concat([df, newCols], axis=1)

    #builds hamming distance map
    hammMatrix, hammGenerations, hammMatrixMIN, hammMatrixMAX = build_hamming_distance_map(df, rows, cols, False)
    overlayMatrix, taskNames = build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []
    
    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(hammGenerations):
        isLastGen = (g_idx == len(hammGenerations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_generation_map(
                hammMatrix[g_idx], overlayMatrix, taskNames,
                0, 1, gen, taskColors, "Avg hamming distance to neighbors",
                figSize=(8,8))
            frames.append(frame)
        
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(hammGenerations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "hammingDistance_fromNeighbors.gif")
    imageio.mimsave(output_path, frames, duration=frameDuration)  # frameDuration ms por frame
    print(f"GIF salved in: {output_path}")

def print_directional_hammming_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskMap:dict, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the hammming distance of a cell to its neighbors, indicating to which cell theyre the most similar to
    Returns gif to the folder given by logdir."""

    dirHammMatrix, dirHammGenerations = build_directional_hamming_map(df, rows, cols, False)
    fitnessMatrix, generations, minmaxDict = build_fitness_map(df, taskMap, rows, cols)
    overlayMatrix, taskNames = build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []

    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(dirHammGenerations):
        isLastGen = (g_idx == len(dirHammGenerations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_hamming_direction_map(
                fitnessMatrix=fitnessMatrix[g_idx], fitMinValue=0, fitMaxValue=1,
                directionalMatrix=dirHammMatrix[g_idx], 
                taskMatrix=overlayMatrix, taskNames=taskNames,
                gen=gen, taskColors=taskColors, legendText="Proportional Fitness",
                figSize=(8,8))
            frames.append(frame)
        
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(hammGenerations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "directionalHammingDistance_fromNeighbors.gif")
    imageio.mimsave(output_path, frames, duration=frameDuration)  # frameDuration ms por frame
    print(f"GIF salved in: {output_path}")

def print_fitness_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskMap:dict, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the fitness of the bot in its own task.
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(characterize_bot).apply(pd.Series)
    # df = pd.concat([df, newCols], axis=1)

    #builds fitness map
    matrix, generations, minmaxDict = build_fitness_map(df, taskMap, rows, cols)
    overlayMatrix, taskNames = build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []

    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(generations):
        isLastGen = (g_idx == len(generations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_generation_map(
                matrix[g_idx], overlayMatrix, taskNames,
                0, 1, gen, taskColors, "Fitness on local task",
                figSize=(8,8))
            frames.append(frame)
            
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(generations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "heatmap_fitness.gif")
    imageio.mimsave(output_path, frames, duration=frameDuration)  # frameDuration ms por frame
    print(f"GIF salved in: {output_path}")

def get_moore_neighbors(pos: tuple[int,int], rows:int, cols:int, toroidal:bool) -> list[tuple[int,int]]:
    """Returns moore neighbors depending of [toroidal] value"""
    x = pos[0]
    y = pos[1]
    output:list[tuple[int,int]] = []
    #Consider toroidal neighbors
    if toroidal:
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0: continue
                # Apply Toroidal effect
                neighPosX = (x + i) % rows
                neighPosY = (y + j) % cols
                output.append((neighPosX,neighPosY))
        return output
    else:
        #Not consider toroidal neighbors
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                if i==0 and j==0: continue
                neighPosX, neighPosY = x+i, y+j
                if (neighPosX < 0 or neighPosY < 0): continue
                elif (neighPosX > rows-1 or neighPosY > cols-1): continue
                output.append((neighPosX,neighPosY))  
        return output

def hamming_distance(shape1: list, shape2: list) -> float:
    A = np.array(shape1).flatten()
    B = np.array(shape2).flatten()
    
    maxDist = max(A.size, B.size) #A and B MUST have the same size!
    dist = np.sum(A != B)
    return dist/maxDist

def simulate_bot(paramDir:str, task:str, shape:list) -> float:
    #load parameters
    with open(os.path.join(paramDir, 'parameters.json'), 'r') as f:
        params = json.load(f)  

    botType = params["robot_type"]
    simSteps = params["sim_step"]
    botShape = np.array(shape)

    #load modules
    robotModule = importlib.import_module(f"robot.{botType}")
    worldModule = importlib.import_module(f"world.{task}")

    #make bot and world
    bot = robotModule.SinRobot()
    bot.shape = botShape
    world = worldModule.get_world()

    #simulate
    world.set_robot(bot)
    world.reset()
    for step in range(simSteps):
        world.step()
    return world.get_score()



def print_line_graph(data:dict, logdir:str, 
                     title:str="Title", 
                     xLabel:str="X", 
                     yLabel:str="Y", 
                     figsize:tuple=(8,8),
                     colors:list[str]=["red","blue","purple","green","pink","gray","black"]):
    """
    Renders a line chart from a list of datasets.

    Each dataset dict must follow this format:
        {
            "label": str,           # legend label
            "x": list[int],         # x-axis values (generations)
            "y": list[float],       # y-axis values (e.g. mean fitness)
            "std": list[float],     # (optional) std deviation for shaded band
        }
    """
    outputPath = os.path.join(logdir, "Graphs")
    os.makedirs(outputPath, exist_ok=True)

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    for idx, (taskName, ds) in enumerate(data.items()):      
        color = colors[idx]
        x = ds["x"]
        y = ds["y"]
        std = ds["std"]
        label = ds.get("label", f"missingLabel{idx}")
        
        ax.plot(x, y, color=color, label=label, linewidth=2)

        yArr = np.array(y)
        stdArr = np.array(std)
        ax.fill_between(x, 
                        yArr-stdArr,
                        yArr+stdArr,
                        color=color,
                        alpha=0.2,
                        linewidth=0,
                        )
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel(xLabel,fontsize=12)
        ax.set_ylabel(yLabel,fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=1)
        ax.set_xlim(0, max(ds["x"][-1] for _, ds in data.items()))
        ax.set_ylim(0, 1)

        fig.tight_layout()
        fig.savefig(f"{outputPath}/{title}", dpi=150)
    plt.close(fig)

def print_scatter_graph(data: dict, logdir: str,
                        title:str="Title", 
                        xLabel:str="X", 
                        yLabel:str="Y", 
                        figsize:tuple=(8,8),
                        colors:list[str]=["red","blue","purple","pink"]):
    
    outputPath = os.path.join(logdir, "Graphs")
    os.makedirs(outputPath, exist_ok=True)
    #save data
    savedDictPath = os.path.join(outputPath, "scatterData.jsonl")
    with open(savedDictPath, "w") as file:
        for bot in data["bots"]:
            record = {**bot, "pos": list(bot["pos"])}  # tuple → list for JSON
            json.dump(record, file)
            file.write("\n")

    plt.close("all")
    fig, ax = plt.subplots(figsize=figsize)

    for i, task in enumerate(data["tasks"]):
        botsFromTask = [bot for bot in data["bots"] if bot["localTask"]==task]
        ax.scatter(
            [bot[data["tasks"][0]] for bot in botsFromTask],
            [bot[data["tasks"][1]] for bot in botsFromTask],
            color = colors[i],
            label = task.split(".")[-1],
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
            s=80,
            zorder=3,
        )
    
        # Diagonal: perfect generalist sits here
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray",
            linewidth=1, label="Same fitness line", zorder=2)

    ax.set_xlabel(f"Normalized fitness — {data['tasks'][0]}", fontsize=11)
    ax.set_ylabel(f"Normalized fitness — {data['tasks'][1]}", fontsize=11)
    ax.set_title("Robots' normalized fitness for both tasks (last generation)", fontsize=13)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(np.arange(0, 1.0, 0.1))
    ax.set_yticks(np.arange(0, 1.0, 0.1))
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, linestyle="--", alpha=1)

    fig.tight_layout()
    fig.savefig(os.path.join(outputPath, "specialization_scatter.png"), dpi=150)
    plt.close(fig)
    print("Scatter saved.")

def plot_hist_data(df:pd.DataFrame, x:str, graphName:str, isDiscrete:bool=True, binsNumber=None):
    """dataframe must have "x" column that will be the value plotted."""   
    plt.close()

    histPlotArgs = {
        "data":df, 
        "x":x, 
        "hue":"experiment", 
        "multiple":"dodge", 
        "discrete": isDiscrete,
        "shrink":0.8, 
        "palette":"muted",
        "edgecolor":'black'
        }

    if not (isDiscrete):
        histPlotArgs["bins"] = binsNumber
    sns.histplot(**histPlotArgs)
    
    existingValues = sorted(df[x].unique())
    plt.xticks(existingValues)


    plt.grid(True, linestyle='--', alpha=0.8) # Grid pontilhado e leve
    plt.ylabel("Frequency")
    plt.savefig(f"Analysis/{graphName}.png", dpi=300, bbox_inches='tight')

def put_data_together(rootLog:str):
    stime = time.time()

    logdirs = []
    allData = []
    for execution in rootLog.iterdir():
        if "parquet" in str(execution): continue
        logdirs.append(str(execution))
        executionName = str(execution).split("log/")[1]
        experimentName = executionName.split("_seed")[0]
        seed = executionName.split("_seed")[1]
        seed = seed.split("_")[0]
        lastGen = -1

        jsonPath = os.path.join(execution,"robots_log.jsonl")
        with open(jsonPath, 'r') as file:
            data = list(file)

        for _, json_str in enumerate(data):
            line = json.loads(json_str)
            line["address"] = str(execution)
            line["experiment"] = experimentName
            line["seed"] = seed

            #characterize bot
            shape = np.array(line["shape"])
            maxBlocks = shape.shape[0] * shape.shape[1]

            ### blocks qty
            line["empty"]     = float(np.sum(shape == 0) / maxBlocks)
            line["rigid"]     = float(np.sum(shape == 1) / maxBlocks)
            line["soft"]      = float(np.sum(shape == 2) / maxBlocks)
            line["h_act"]     = float(np.sum(shape == 3) / maxBlocks)
            line["v_act"]     = float(np.sum(shape == 4) / maxBlocks)
            line["actuators"] = float(line["h_act"] + line["v_act"])
            
            ### bot height and width
            notZero = np.argwhere(shape)     #gets all positions that are NOT zero
            minY, minX = notZero.min(axis=0) #min position that is not zero
            maxY, maxX = notZero.max(axis=0) #max position that is not zero
            line["height"] = int(maxY - minY + 1)
            line["width"] = int(maxX - minX + 1)
            
            ### get simmetry
            horizontalMirror = np.fliplr(shape)
            verticalMirror = np.flipud(shape)
            line["hSimScore"] = float(np.sum(shape == horizontalMirror) / shape.size)
            line["vSimScore"] = float(np.sum(shape == verticalMirror) / shape.size)
            
            if (line["gen"] > lastGen) and (line["gen"]!=99999): lastGen = line["gen"]
            #Last gen all bots are simulated in all task they werent to get their fitness. these bots that were simulated get gen=99999 to highlight them
            #this part puts this fitness into the fitness of this same bot but in the last identified gen
            if line["gen"] == 99999: 
                for data in allData:
                    if (data["address"] == line["address"]) and (data["gen"]==lastGen) and (data["id"]==line["id"]):
                        data["fit"] = line["fit"]
                        break
                continue
            allData.append(line)

    end1=time.time()    
    print(f"Finished reading all data {end1-stime} seconds.")
    df = pd.DataFrame(allData)
    end2=time.time()    
    print(f"Finished pandasFile {end2-end1} seconds.")

    fitCols = pd.json_normalize(df['fit']).add_prefix('fit_')
    df = pd.concat([df.drop(columns=['fit']), fitCols], axis=1)
    df.to_parquet(os.path.join(rootLog,f"completeData.parquet"), engine='pyarrow', index=False)
    end3=time.time()    
    print(f"Finished write parquet file in {end3-end2} seconds.")

def build_hist_data(df:pd.DataFrame, wantedData:str, fitMin:float, experiments:list, fitNames:list, taskMap:list):
    dataframes = []
       
    for expName in experiments:
        for fitName in fitNames:
            mask = (df["experiment"] == expName) & (df[fitName] > fitMin)
            new_df = df.loc[mask, ["experiment", wantedData]].copy()
            dataframes.append(new_df)
    
    output_df = pd.concat(dataframes, ignore_index=True)
    return output_df

def evaluate_bots_from_archive(df:pd.DataFrame):
    """df is arquive created by put_data_together, baselines were not evaluated in other task."""
    df = df.copy()
    gen = 500
    info = [
        ("baseline-walkerv0", "log/baseline-walkerv0_seed7_CGA_03271207", "BridgeWalker_v0", "fit_world.BridgeWalker_v0"),
        ("baseline-BridgeWalker_v0", "log/baseline-BridgeWalker_v0_seed7_CGA_03272353", "Walker_v0", "fit_world.Walker_v0")
    ]

    for experiment, param, evaluatedTask, fitName in info:
        mask1 = (df["experiment"]==experiment) & (df["gen"]==gen) 
        shapes = df.loc[mask1, "shape"].copy()

        alreadyFilled = df.loc[mask1, fitName].notna().sum()
        assert alreadyFilled == 0, (
            f"{alreadyFilled} lines in '{fitName}' already have a value for {experiment}. "
        )

        scores = []
        for shape in shapes:
            shape = [matrix.tolist() for matrix in shape]
            score = simulate_bot(paramDir=param, task=evaluatedTask, shape=shape)
            scores.append(score)
        df.loc[mask1, fitName] = scores

    df.to_parquet(os.path.join("log",f"completeData_evalBaselines.parquet"), engine='pyarrow', index=False)


if __name__=="__main__":
    # rootLog = "log"
    # rootLog = pathlib.Path(rootLog)
    
    # # put_data_together(rootLog=rootLog)
    # # df = pd.read_parquet("log/completeData_evalBaselines.parquet")
    # # evaluate_bots_from_archive(df)
    # df, fitNames, minMaxValues = load_parquet_log("log/completeData_evalBaselines.parquet")

    # ### All columns of DF
    # columns = list(df.columns)
    # #['id', 'gen', 'pos', 'parent2', 'shape', 'address', 'experiment', 
    # # 'seed', 'empty', 'rigid', 'soft', 'h_act', 'v_act', 'actuators', 
    # # 'height', 'width', 'hSimScore', 'vSimScore', 'fit_world.Walker_v0', 
    # # 'fit_world.BridgeWalker_v0']
    # ### All experiments in DF
    # experiments = list(df["experiment"].unique())
    # # ['baseline-walkerv0', 'quadrant-v0_childFirst', 
    # #  'quadrantv0', 'baseline-BridgeWalker_v0']
    # ### Fit names and max fitness found
    # fitNames = ["fit_world.Walker_v0", "fit_world.BridgeWalker_v0"]
    # maxFits = df.groupby("experiment")[fitNames].max()
    # # Max fits found:                           fit_world.Walker_v0  fit_world.BridgeWalker_v0
    # # experiment                                                              
    # # baseline-BridgeWalker_v0                  NaN                   0.981070
    # # baseline-walkerv0                    0.932207                        NaN
    # # quadrant-v0_childFirst               1.000000                   1.000000
    # # quadrantv0                           0.923777                   0.924497
    # ### Getting task map (its the same for all quadrants)
    # taskmap_path = os.path.join("log/quadrant-v0_childFirst_seed7_CGA_04021805", "grid_taskMap.json")
    # with open(taskmap_path, "r") as f:
    #     taskMap = json.load(f)   

    # experiments = ["baseline-walkerv0", "baseline-BridgeWalker_v0", "quadrantv0"]

    # #v_act and h_act
    # wantedDatas = ["v_act", "h_act"]
    # for wantedData in wantedDatas:
    #     fitsMin = [0,0.5,0.75,0.85,0.9,0.95]
    #     for fitMin in fitsMin:
    #         new_df, maxFits = build_hist_data(df, wantedData, fitMin, experiments, fitNames)
    #         new_df[wantedData] = new_df[wantedData] * 25
    #         plot_hist_data(new_df, x=wantedData, graphName=f"{wantedData}-Comparison of baseline and quadrand tasks (fit>{fitMin})", isDiscrete=True)
        
    # # #height, width, 
    # # wantedDatas = ["height", "width"]
    # # for wantedData in wantedDatas:
    # #     fitsMin = [0]
    # #     for fitMin in fitsMin:
    # #         new_df, maxFits = build_hist_data(df, wantedData, fitMin, experiments, fitNames)
    # #         plot_hist_data(new_df, x=wantedData, graphName=f"{wantedData}-Comparison of baseline tasks (fit>{fitMin})", isDiscrete=True)

    # # # simmetry
    # # wantedDatas = ["vSimScore", "hSimScore"]
    # # for wantedData in wantedDatas:
    # #     fitsMin = [0, 0.5, 0.75, 0.85, 0.9, 0.95]
    # #     for fitMin in fitsMin:
    # #         new_df, maxFits = build_hist_data(df, wantedData, fitMin, experiments, fitNames)
    # #         new_df[wantedData] = new_df[wantedData].round(1).astype(str)
    # #         plot_hist_data(new_df, x=wantedData, graphName=f"{wantedData}-Comparison of baseline tasks (fit>{fitMin})", isDiscrete=True)

















#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
#############################################################################
    #adds columns describing each robot


    # logdirs = []
    # for execution in rootLog.iterdir():
    #     logdirs.append(str(execution))
    #     # executionName = str(execution).split("log/")[1]
    #     # experimentName = executionName.split("_seed")[0]
    #     # seed = executionName.split("_seed")[1]
    #     # seed = seed.split("_")[0]

    logdirs = ["log/quadrantv1_seed7_CGA_04302108", "log/baseline-BridgeWalkerv1_seed7_CGA_04291045", "log/baseline-walkerv1_seed7_CGA_04281958"]

    for i, logdir in enumerate(logdirs):
        #prepare dfs
        df, taskMap, gridSize = load_log(logdir)
        rows, cols = gridSize
        #adds columns describing each robot
        # newCols = df["shape"].apply(characterize_bot).apply(pd.Series)
        # df = pd.concat([df, newCols], axis=1)
        
    #     #get images
    #     print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(0,0))
    #     # print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(1,0))
    #     # print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(4,2))
    #     # print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(4,3))
    #     # print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(5,3))
    #     # print_bot(logdir=logdir, df=df, rows=rows, cols=cols, gen=500, pos=(4,2))

    #     # data = build_fitness_data(df=df, taskMap=taskMap)
    #     # data2 = build_hamming_data(df=df, taskMap=taskMap)
    #     data3 = build_fit_scatter_data (df=df, taskMap=taskMap)

    #     # print_line_graph(data=data, logdir=logdir, 
    #     #                  title="Average proportional fitness value (related to maximum found in each task)", 
    #     #                  xLabel="Generation", 
    #     #                  yLabel="Avg. proportional fitness",
    #     #                  )

    #     # print_line_graph(data=data2, logdir=logdir, 
    #     #                  title="Average hamming global and per task hamming distance", 
    #     #                  xLabel="Generation", 
    #     #                  yLabel="Avg. Hamming distance",
    #                     #  )
        
    #     print_scatter_graph(data=data3, logdir=logdir,
    #                     title="Title", 
    #                     xLabel="X", 
    #                     yLabel="Y", 
    #                     figsize=(8,8),
    #                     colors=["red","blue","purple","pink"])

        # print_hammming_map_gif(logdir=logdir, df=df, rows= rows, cols=cols, taskMap=taskMap, taskColors=["green","purple"], frameInterval=5, frameDuration=300)
        # print_fitness_map_gif(logdir=logdir, df=df, rows= rows, cols=cols, taskMap=taskMap, taskColors=["green","purple"], frameInterval=5, frameDuration=300)
        print_directional_hammming_map_gif(logdir=logdir, df=df, rows= rows, cols=cols, taskMap=taskMap, taskColors=["green","purple"], frameInterval=5, frameDuration=300)
    # #---------------------------










