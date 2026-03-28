import pandas as pd, numpy as np
import json, os, imageio
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
    realGens = df[df["gen"] != 99999]["gen"].unique()
    lastRealGen = max(realGens)
    extraEvals = df[df["gen"] == 99999][["id", "fit", "pos"]].set_index("id")

    for botId, extraFit in extraEvals["fit"].items():
        mask = (df["id"] == botId) & (df["gen"] == lastRealGen)
        if mask.any():
            df.loc[mask, "fit"] = df.loc[mask, "fit"].apply(lambda x: {**x, **extraFit})
    
    df = df[df["gen"] != 99999]
    return df, taskMap, (rows, cols)

def count_blocks(shape: list) -> dict:
    shape = np.array(shape)
    blocks = {}
    blocks["empty"]     = int(np.sum(shape == 0))
    blocks["rigid"]     = int(np.sum(shape == 1))
    blocks["soft"]      = int(np.sum(shape == 2))
    blocks["h_act"]     = int(np.sum(shape == 3))
    blocks["v_act"]     = int(np.sum(shape == 4))
    blocks["actuators"] = blocks["h_act"] + blocks["v_act"]
    blocks["n_blocks"]  = blocks["rigid"] + blocks["soft"] + blocks["h_act"] + blocks["v_act"]
    return blocks

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

def print_actuators_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the amount of actuators in each robot.
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(count_blocks).apply(pd.Series)
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

def print_hammming_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the hammming distance of a cell to its neighbors
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(count_blocks).apply(pd.Series)
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

def print_fitness_map_gif(logdir:str, df:pd.DataFrame, rows:int, cols:int, taskColors:str, frameInterval:int=5, frameDuration:float=300):
    """
    Generates gif of all generation's heatmap considering the fitness of the bot in its own task.
    Returns gif to the folder given by logdir."""

    # #creates dataframe
    # df, taskMap, gridSize = load_log(logdir)
    # rows, cols = gridSize

    # #adds columns describing each robot
    # newCols = df["shape"].apply(count_blocks).apply(pd.Series)
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



if __name__=="__main__":
    logdirs = [
        # "log/baseline-walkerv0_seed7_CGA_03271207",
        # "log/baseline-walkerv0_seed49_CGA_03271327",
        # "log/baseline-walkerv0_seed343_CGA_03271447",
        # "log/baseline-walkerv0_seed2401_CGA_03271611",
        # "log/baseline-walkerv0_seed16807_CGA_03271729",
        # "log/baseline-BridgeWalker_v0_seed7_CGA_03272353",
        # "log/baseline-BridgeWalker_v0_seed49_CGA_03280249",
        # "log/baseline-BridgeWalker_v0_seed343_CGA_03280548",
        # "log/baseline-BridgeWalker_v0_seed2401_CGA_03280851",
        # "log/baseline-BridgeWalker_v0_seed16807_CGA_03281201",
        "log/quadrant-BridgeWalker_v0_seed7_CGA_03281508"
    ]

    for i, logdir in enumerate(logdirs):
        #prepare dfs
        df, taskMap, gridSize = load_log(logdir)
        rows, cols = gridSize
        #adds columns describing each robot
        newCols = df["shape"].apply(count_blocks).apply(pd.Series)
        df = pd.concat([df, newCols], axis=1)
        
        #get images
        print_hammming_map_gif(logdir=logdir, df=df, rows= rows, cols=cols, taskColors=["black","green"], frameInterval=100, frameDuration=300)
        print_fitness_map_gif(logdir=logdir, df=df, rows= rows, cols=cols, taskColors=["black","green"], frameInterval=100, frameDuration=300)



    # for logdir in logdirs:
        # # print_actuators_map_gif(logdir=logdir, taskColors=["black","green"])
        # print_hammming_map_gif(logdir=logdir, taskColors=["black","green"], frameInterval=5)
        # print_fitness_map_gif(logdir=logdir, taskColors=["black","green"], frameInterval=5)

    #---------------------------
    # df, taskMap, gridSize = load_log(logdirs[0])

    # rows, cols = gridSize
    # build_fitness_map(df, taskMap, rows, cols)

    # newCols = df["shape"].apply(count_blocks).apply(pd.Series)
    # df = pd.concat([df, newCols], axis=1)
    # globalHammMatrix, globalHammGenerations, globalHammMatrixMIN, globalHammMatrixMAX = build_global_hamming_distance_map(df, rows, cols)

    # # hammMatrix, hammGenerations, hammMatrixMIN, hammMatrixMAX = build_hamming_distance_map(df, rows, cols, False)
    # # print(hammMatrix[99])



