import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis as analysis
import data.loaders as loaders
import seaborn as sns
import pandas as pd, numpy as np
import json, imageio, importlib, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pygifsicle import optimize
from pathlib import Path

TASKMAPS = {
    'baseline-walkerv1': {"(0,0)":"world.Walker_v0","(1,0)":"world.Walker_v0","(2,0)":"world.Walker_v0","(3,0)":"world.Walker_v0","(4,0)":"world.Walker_v0","(5,0)":"world.Walker_v0","(6,0)":"world.Walker_v0","(7,0)":"world.Walker_v0","(8,0)":"world.Walker_v0","(9,0)":"world.Walker_v0","(0,1)":"world.Walker_v0","(1,1)":"world.Walker_v0","(2,1)":"world.Walker_v0","(3,1)":"world.Walker_v0","(4,1)":"world.Walker_v0","(5,1)":"world.Walker_v0","(6,1)":"world.Walker_v0","(7,1)":"world.Walker_v0","(8,1)":"world.Walker_v0","(9,1)":"world.Walker_v0","(0,2)":"world.Walker_v0","(1,2)":"world.Walker_v0","(2,2)":"world.Walker_v0","(3,2)":"world.Walker_v0","(4,2)":"world.Walker_v0","(5,2)":"world.Walker_v0","(6,2)":"world.Walker_v0","(7,2)":"world.Walker_v0","(8,2)":"world.Walker_v0","(9,2)":"world.Walker_v0","(0,3)":"world.Walker_v0","(1,3)":"world.Walker_v0","(2,3)":"world.Walker_v0","(3,3)":"world.Walker_v0","(4,3)":"world.Walker_v0","(5,3)":"world.Walker_v0","(6,3)":"world.Walker_v0","(7,3)":"world.Walker_v0","(8,3)":"world.Walker_v0","(9,3)":"world.Walker_v0","(0,4)":"world.Walker_v0","(1,4)":"world.Walker_v0","(2,4)":"world.Walker_v0","(3,4)":"world.Walker_v0","(4,4)":"world.Walker_v0","(5,4)":"world.Walker_v0","(6,4)":"world.Walker_v0","(7,4)":"world.Walker_v0","(8,4)":"world.Walker_v0","(9,4)":"world.Walker_v0","(0,5)":"world.Walker_v0","(1,5)":"world.Walker_v0","(2,5)":"world.Walker_v0","(3,5)":"world.Walker_v0","(4,5)":"world.Walker_v0","(5,5)":"world.Walker_v0","(6,5)":"world.Walker_v0","(7,5)":"world.Walker_v0","(8,5)":"world.Walker_v0","(9,5)":"world.Walker_v0","(0,6)":"world.Walker_v0","(1,6)":"world.Walker_v0","(2,6)":"world.Walker_v0","(3,6)":"world.Walker_v0","(4,6)":"world.Walker_v0","(5,6)":"world.Walker_v0","(6,6)":"world.Walker_v0","(7,6)":"world.Walker_v0","(8,6)":"world.Walker_v0","(9,6)":"world.Walker_v0","(0,7)":"world.Walker_v0","(1,7)":"world.Walker_v0","(2,7)":"world.Walker_v0","(3,7)":"world.Walker_v0","(4,7)":"world.Walker_v0","(5,7)":"world.Walker_v0","(6,7)":"world.Walker_v0","(7,7)":"world.Walker_v0","(8,7)":"world.Walker_v0","(9,7)":"world.Walker_v0","(0,8)":"world.Walker_v0","(1,8)":"world.Walker_v0","(2,8)":"world.Walker_v0","(3,8)":"world.Walker_v0","(4,8)":"world.Walker_v0","(5,8)":"world.Walker_v0","(6,8)":"world.Walker_v0","(7,8)":"world.Walker_v0","(8,8)":"world.Walker_v0","(9,8)":"world.Walker_v0","(0,9)":"world.Walker_v0","(1,9)":"world.Walker_v0","(2,9)":"world.Walker_v0","(3,9)":"world.Walker_v0","(4,9)":"world.Walker_v0","(5,9)":"world.Walker_v0","(6,9)":"world.Walker_v0","(7,9)":"world.Walker_v0","(8,9)":"world.Walker_v0","(9,9)":"world.Walker_v0"},
    'quadrantv1': {"(0,0)":"world.Walker_v0","(1,0)":"world.Walker_v0","(2,0)":"world.Walker_v0","(3,0)":"world.Walker_v0","(4,0)":"world.Walker_v0","(5,0)":"world.BridgeWalker_v0","(6,0)":"world.BridgeWalker_v0","(7,0)":"world.BridgeWalker_v0","(8,0)":"world.BridgeWalker_v0","(9,0)":"world.BridgeWalker_v0","(0,1)":"world.Walker_v0","(1,1)":"world.Walker_v0","(2,1)":"world.Walker_v0","(3,1)":"world.Walker_v0","(4,1)":"world.Walker_v0","(5,1)":"world.BridgeWalker_v0","(6,1)":"world.BridgeWalker_v0","(7,1)":"world.BridgeWalker_v0","(8,1)":"world.BridgeWalker_v0","(9,1)":"world.BridgeWalker_v0","(0,2)":"world.Walker_v0","(1,2)":"world.Walker_v0","(2,2)":"world.Walker_v0","(3,2)":"world.Walker_v0","(4,2)":"world.Walker_v0","(5,2)":"world.BridgeWalker_v0","(6,2)":"world.BridgeWalker_v0","(7,2)":"world.BridgeWalker_v0","(8,2)":"world.BridgeWalker_v0","(9,2)":"world.BridgeWalker_v0","(0,3)":"world.Walker_v0","(1,3)":"world.Walker_v0","(2,3)":"world.Walker_v0","(3,3)":"world.Walker_v0","(4,3)":"world.Walker_v0","(5,3)":"world.BridgeWalker_v0","(6,3)":"world.BridgeWalker_v0","(7,3)":"world.BridgeWalker_v0","(8,3)":"world.BridgeWalker_v0","(9,3)":"world.BridgeWalker_v0","(0,4)":"world.Walker_v0","(1,4)":"world.Walker_v0","(2,4)":"world.Walker_v0","(3,4)":"world.Walker_v0","(4,4)":"world.Walker_v0","(5,4)":"world.BridgeWalker_v0","(6,4)":"world.BridgeWalker_v0","(7,4)":"world.BridgeWalker_v0","(8,4)":"world.BridgeWalker_v0","(9,4)":"world.BridgeWalker_v0","(0,5)":"world.BridgeWalker_v0","(1,5)":"world.BridgeWalker_v0","(2,5)":"world.BridgeWalker_v0","(3,5)":"world.BridgeWalker_v0","(4,5)":"world.BridgeWalker_v0","(5,5)":"world.Walker_v0","(6,5)":"world.Walker_v0","(7,5)":"world.Walker_v0","(8,5)":"world.Walker_v0","(9,5)":"world.Walker_v0","(0,6)":"world.BridgeWalker_v0","(1,6)":"world.BridgeWalker_v0","(2,6)":"world.BridgeWalker_v0","(3,6)":"world.BridgeWalker_v0","(4,6)":"world.BridgeWalker_v0","(5,6)":"world.Walker_v0","(6,6)":"world.Walker_v0","(7,6)":"world.Walker_v0","(8,6)":"world.Walker_v0","(9,6)":"world.Walker_v0","(0,7)":"world.BridgeWalker_v0","(1,7)":"world.BridgeWalker_v0","(2,7)":"world.BridgeWalker_v0","(3,7)":"world.BridgeWalker_v0","(4,7)":"world.BridgeWalker_v0","(5,7)":"world.Walker_v0","(6,7)":"world.Walker_v0","(7,7)":"world.Walker_v0","(8,7)":"world.Walker_v0","(9,7)":"world.Walker_v0","(0,8)":"world.BridgeWalker_v0","(1,8)":"world.BridgeWalker_v0","(2,8)":"world.BridgeWalker_v0","(3,8)":"world.BridgeWalker_v0","(4,8)":"world.BridgeWalker_v0","(5,8)":"world.Walker_v0","(6,8)":"world.Walker_v0","(7,8)":"world.Walker_v0","(8,8)":"world.Walker_v0","(9,8)":"world.Walker_v0","(0,9)":"world.BridgeWalker_v0","(1,9)":"world.BridgeWalker_v0","(2,9)":"world.BridgeWalker_v0","(3,9)":"world.BridgeWalker_v0","(4,9)":"world.BridgeWalker_v0","(5,9)":"world.Walker_v0","(6,9)":"world.Walker_v0","(7,9)":"world.Walker_v0","(8,9)":"world.Walker_v0","(9,9)":"world.Walker_v0"},
    'baseline-BridgeWalkerv1': {"(0,0)":"world.BridgeWalker_v0","(1,0)":"world.BridgeWalker_v0","(2,0)":"world.BridgeWalker_v0","(3,0)":"world.BridgeWalker_v0","(4,0)":"world.BridgeWalker_v0","(5,0)":"world.BridgeWalker_v0","(6,0)":"world.BridgeWalker_v0","(7,0)":"world.BridgeWalker_v0","(8,0)":"world.BridgeWalker_v0","(9,0)":"world.BridgeWalker_v0","(0,1)":"world.BridgeWalker_v0","(1,1)":"world.BridgeWalker_v0","(2,1)":"world.BridgeWalker_v0","(3,1)":"world.BridgeWalker_v0","(4,1)":"world.BridgeWalker_v0","(5,1)":"world.BridgeWalker_v0","(6,1)":"world.BridgeWalker_v0","(7,1)":"world.BridgeWalker_v0","(8,1)":"world.BridgeWalker_v0","(9,1)":"world.BridgeWalker_v0","(0,2)":"world.BridgeWalker_v0","(1,2)":"world.BridgeWalker_v0","(2,2)":"world.BridgeWalker_v0","(3,2)":"world.BridgeWalker_v0","(4,2)":"world.BridgeWalker_v0","(5,2)":"world.BridgeWalker_v0","(6,2)":"world.BridgeWalker_v0","(7,2)":"world.BridgeWalker_v0","(8,2)":"world.BridgeWalker_v0","(9,2)":"world.BridgeWalker_v0","(0,3)":"world.BridgeWalker_v0","(1,3)":"world.BridgeWalker_v0","(2,3)":"world.BridgeWalker_v0","(3,3)":"world.BridgeWalker_v0","(4,3)":"world.BridgeWalker_v0","(5,3)":"world.BridgeWalker_v0","(6,3)":"world.BridgeWalker_v0","(7,3)":"world.BridgeWalker_v0","(8,3)":"world.BridgeWalker_v0","(9,3)":"world.BridgeWalker_v0","(0,4)":"world.BridgeWalker_v0","(1,4)":"world.BridgeWalker_v0","(2,4)":"world.BridgeWalker_v0","(3,4)":"world.BridgeWalker_v0","(4,4)":"world.BridgeWalker_v0","(5,4)":"world.BridgeWalker_v0","(6,4)":"world.BridgeWalker_v0","(7,4)":"world.BridgeWalker_v0","(8,4)":"world.BridgeWalker_v0","(9,4)":"world.BridgeWalker_v0","(0,5)":"world.BridgeWalker_v0","(1,5)":"world.BridgeWalker_v0","(2,5)":"world.BridgeWalker_v0","(3,5)":"world.BridgeWalker_v0","(4,5)":"world.BridgeWalker_v0","(5,5)":"world.BridgeWalker_v0","(6,5)":"world.BridgeWalker_v0","(7,5)":"world.BridgeWalker_v0","(8,5)":"world.BridgeWalker_v0","(9,5)":"world.BridgeWalker_v0","(0,6)":"world.BridgeWalker_v0","(1,6)":"world.BridgeWalker_v0","(2,6)":"world.BridgeWalker_v0","(3,6)":"world.BridgeWalker_v0","(4,6)":"world.BridgeWalker_v0","(5,6)":"world.BridgeWalker_v0","(6,6)":"world.BridgeWalker_v0","(7,6)":"world.BridgeWalker_v0","(8,6)":"world.BridgeWalker_v0","(9,6)":"world.BridgeWalker_v0","(0,7)":"world.BridgeWalker_v0","(1,7)":"world.BridgeWalker_v0","(2,7)":"world.BridgeWalker_v0","(3,7)":"world.BridgeWalker_v0","(4,7)":"world.BridgeWalker_v0","(5,7)":"world.BridgeWalker_v0","(6,7)":"world.BridgeWalker_v0","(7,7)":"world.BridgeWalker_v0","(8,7)":"world.BridgeWalker_v0","(9,7)":"world.BridgeWalker_v0","(0,8)":"world.BridgeWalker_v0","(1,8)":"world.BridgeWalker_v0","(2,8)":"world.BridgeWalker_v0","(3,8)":"world.BridgeWalker_v0","(4,8)":"world.BridgeWalker_v0","(5,8)":"world.BridgeWalker_v0","(6,8)":"world.BridgeWalker_v0","(7,8)":"world.BridgeWalker_v0","(8,8)":"world.BridgeWalker_v0","(9,8)":"world.BridgeWalker_v0","(0,9)":"world.BridgeWalker_v0","(1,9)":"world.BridgeWalker_v0","(2,9)":"world.BridgeWalker_v0","(3,9)":"world.BridgeWalker_v0","(4,9)":"world.BridgeWalker_v0","(5,9)":"world.BridgeWalker_v0","(6,9)":"world.BridgeWalker_v0","(7,9)":"world.BridgeWalker_v0","(8,9)":"world.BridgeWalker_v0","(9,9)":"world.BridgeWalker_v0"},
}

COLOR_MAP = {
    #experiment name           ,  task name               :  #color     ,  label
    ('baseline-walkerv1',        'world.Walker_v0')       : ("#000000", 'Baseline Walker'),
    ('baseline-BridgeWalkerv1', 'world.BridgeWalker_v0') : ("#000000", 'Baseline Bridge'),
    ('quadrantv1',               'world.Walker_v0')       : ("#FF0000", 'Quadrant Walker'),
    ('quadrantv1',               'world.BridgeWalker_v0') : ("#FF0000", 'Quadrant Bridge'),
}

CHARACTERISTICS = ['empty', 'rigid', 'soft', 'h_act', 'v_act', 'height', 'width', 'hSimScore', 'vSimScore']

def get_task_for_robot(row):
    """Returns task where the bot was optimized given its position and TASKMAP."""
    experiment = row['experiment']
    x, y = row['pos']
    pos = f"({x},{y})"
    return TASKMAPS[experiment][pos]

def get_fitness_of_assigned_task(row):
    """returns fitness value where the bot was optimized on.
    *** requires "assignedTask" column, can be obtained with [get_task_for_robot] function """
    newCol = 'fit_' + row['assignedTask']   # ex: 'fit_world.Walker_v0'
    return row[newCol]

"""
1) Robot characteristics analysis.
    - The idea here is to analyse robots variables (type of blocks, simmetry, etc) to see if different tasks produce different types of robots.
    - So. First:
        - Plot a scatter plot where X is the value of a variable and Y is the value of another variable, each point is a robot. Each color is a task of a map.
        - This will show me all the variables combinations and what type of robot is produced by each grid environment.
    - Second:
        - For the maps where things look more sparce, where there's visible distance between points of different colors I will deeply analyse it.
        - Deep analysis is divide the space in bins and count how many where filled for each point of each color.
    - Third:
        - Check how good all robots are in a task they were not trained on.
"""
def build_pair_data(df:pd.DataFrame, minFit:float):
    """
    Edits DF with columns to be used by print_pair_data
    - 'assignedTask': task where bot was optimized on
    - 'assignedFit': fitness value where bot was optimized on
    - 'colorKey': unique color to point in pair plot
    - 'colorLabel': legend label
    """
    df = df.loc[df['experiment'] != 'quadrant-v0_childFirst'].copy()
    df['assignedTask'] = df.apply(get_task_for_robot, axis=1)  
    df['assignedFit'] = df.apply(get_fitness_of_assigned_task, axis=1)
    df = df.loc[df['assignedFit'] >= minFit].copy()
    df = df.sort_values('gen')
    df = df.drop_duplicates(subset=['id', 'seed', 'experiment'], keep='last')

    df['colorKey'] = list(zip(df['experiment'], df['assignedTask'])) #zip puts together values in a tuple. ex: output[0] = (experiment[0], assignedTask[0])
    labels = [] #make a list of labels and colors
    for _, row in df.iterrows():
        colorKey = (row['experiment'], row['assignedTask'])
        label = COLOR_MAP[colorKey][1]   # [1] is label
        labels.append(label)
    df['colorLabel'] = labels
    return df

def print_pair_plot(archivePath:str, minFit:float=0.8):
    """
    Uses output from [build_pair_data] to print a pair plot of characteristics of bots that have a minimum fitness.
    minFit here must be the same used in [build_pair_data]
    """
    df, _, _ = loaders.load_parquet_log(archivePath)
    logDir= str(Path(archivePath).parent)
    df = build_pair_data(df, minFit)
    columnsNeeded = CHARACTERISTICS + ['colorLabel', 'assignedTask']
    df = df[columnsNeeded].copy()

    # used in scatter kind
    noise = 0.01 #all points were on each other
        
    #colors by unique task of each experiment
    labelColors = {}
    for value in COLOR_MAP.values():
        labelColors[value[1]] = value[0]

    #one picture to each task
    for task in df['assignedTask'].unique():
        df_task = df.loc[df['assignedTask'] == task].copy()

        # for col in CHARACTERISTICS:
        #     noiseCol = np.random.uniform(-noise, noise, size=len(df_task))
        #     df_task[col] = df_task[col] + noiseCol

        pair = sns.pairplot(
            df_task,
            vars=CHARACTERISTICS,
            hue='colorLabel',
            palette=labelColors,
            kind='scatter',
            plot_kws={'alpha': 1, 's': 15}, #for kind scatter
            diag_kind='kde',
            # corner=True,
            markers=["s","d"],
        )

        
        for ax in pair.axes.flatten():
            ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)

        # pair.map_lower(sns.kdeplot, levels=4, color=".2")
        pair.figure.suptitle(f'Pairplot por experimento+task  (fitness >= {minFit})', y=1.01)
        pair.figure.savefig(f'{logDir}/noNOisePairplot_{minFit}-{task}.png', bbox_inches='tight', dpi=150)
        plt.close(pair.figure)
        print(f"Salvo: {logDir}/pairplot_{task}.png")
"""
 Goals
 1) Fitness curve of each task along generations.
    - Average and Standard deviation of robot's fitness that were optimized in the same task.
    - 4 Curves
        - baseline-walkerv0
        - baseline-BridgeWalker_v0
        - quadrantv0 (walker)
        - quadrantv0 (bridge)
    - Check:
        - Does quadrant sacrifice fitness to acomodate two tasks? Does quadrand achieve reasonable level of fitness?
"""
def build_fitness_data(df: pd.DataFrame):
    """
    Reads the robots log dataframe and builds one dataset per task per experiment
        - Maps(experiments) that have more than one task would have 2 dicts as output: One for each task
    Each dataset contains the mean fitness and std deviation per generation.

    Returns a list of dicts with format:
        {
            "label": str,        # experiment-task (short description)
            "x": list[int],      # generation indices
            "y": list[float],    # mean fitness per generation (across seed)
            "std": list[float],  # std deviation across seeds (or within itself if only seed)
        }
    """
    df = df.loc[df["experiment"] != "quadrant-v0_childFirst"].copy()
    # lista com todas as gerações
    # para cada geração, passar por cada celular e pegar o valor de fitness da task daquela célula e colocar numa lista.
    # tirar média e desvio padrão
    generations = sorted(df["gen"].unique())
    experiments = sorted(df["experiment"].unique())
    output = {}

    #para cada experimento
    for experiment in experiments:
    #   para cada unique task daquele experimento, um label
        newDf = df.loc[df["experiment"] == experiment].copy()
        uniqueTasks = list(set(TASKMAPS[experiment].values()))

        for i,task in enumerate(uniqueTasks):
            output[f"{experiment}-{task}"] = {}
            output[f"{experiment}-{task}"]["label"] = f"{experiment}-{task}"
            output[f"{experiment}-{task}"]["x"] = []
            output[f"{experiment}-{task}"]["y"] = []
            output[f"{experiment}-{task}"]["std"] = []

        for gen in generations:
            genBots = newDf.loc[newDf["gen"]==gen].copy()
            valueDict = {}

            for _, row in genBots.iterrows():
                x, y = row["pos"]
                pos = f"({x},{y})"
                taskName = TASKMAPS[experiment][pos]
                if taskName not in valueDict: valueDict[taskName] = []
                fitValue = row[f"fit_{taskName}"]
                valueDict[taskName].append(fitValue)
        
            for task in uniqueTasks:
                avgValue = np.mean(valueDict[task])
                stdValue = np.std(valueDict[task])

                output[f"{experiment}-{task}"]["x"].append(gen)
                output[f"{experiment}-{task}"]["y"].append(avgValue)
                output[f"{experiment}-{task}"]["std"].append(stdValue)
    return output   
"""
 2) Hamming distance: Same task X Different task
    - Different task: How similar are the bots that were optimized for different things.
    - Same task: How similar are the bots that were optimized for the same thing.
    - General: Hamming distance between bots independently of the task they were optimized in.
    - Two ideas:
        - Graph1: Line along generations
            - 6 Curves:      
                - baseline-walkerv0 [general] (theres only one task)
                - baseline-BridgeWalker_v0 [general] (theres only one task)
                - quadrantv0 [general]
                - quadrantv0-walker [same task]
                - quadrantv0-bridge [same task]
                - quadrantv0 [different task]  
            - Each point is the average hamming distance of pairs (different task pairs, same task pairs and any pairs)
            - Check: 
                - Graph 1.1: 3 Curves: Comparison between generals -> How diversity evolves
                - Graph 1.2: 2 Curves: Quad Same task + Quad different task 
                    - If same = different, suggest that tasks are similar, robot converged to one form. probably robust one.
                    - If same != different, suggest that tasks requires different robots, no global conversion.
                - Graph 1.3: Bonus: 2 Curves: Comparison of quads that have the same task.
                    - One figure for each task. One Curve for each quadrand.
                    - Compare robots of one quadrand to another quadrand.
                    - Same task but, did evolution happened the same way? How different are the bots that were optimized for the same task but had frontiers.
        - Graph2: BoxPlot of last generation robots.
            - X axis: Each point is a "curve" of Graph1. 
                - baseline-walkerv0 [general], baseline-BridgeWalker_v0 [general], and so on...
            - Y axis: 
                - Hamming distance of last generation in each seed for all X axis points.
            - Check:
                - Consistency of hamming distance along the seeds for the best bots of each case.
                - Same level of analysis of Graph1.
                    - Graph 2.1: Between generals
                    - Graph 2.2: Inside Quads and same task baseline [for walker]
                    - Graph 2.3: Inside Quads and same task baseline [for bridge]
    - Check
        - Are bots that were optimized in the same task similar? 
        - Did the new grid (quadrand) create differences between robots?
"""
def build_generational_hamming_data(df:pd.DataFrame):
    """
    Reads the robots log dataframe and builds one dict for each situation:                
                - baseline-walkerv0 [general] (theres only one task)
                - baseline-BridgeWalker_v0 [general] (theres only one task)
                - quadrantv0 [general]
                - quadrantv0-walker [same task]
                - quadrantv0-bridge [same task]
                - quadrantv0 [different task]  
    Each dataset contains the mean fitness and std deviation per generation.
    Returns a list of dicts with format:
        {
            "label": str,        # experiment-task (short description)
            "x": list[int],      # generation indices
            "y": list[float],    # mean fitness per generation (across seed)
            "std": list[float],  # std deviation across seeds (or within itself if only seed)
        }
    """
    df = df.loc[df["experiment"] != "quadrant-v0_childFirst"].copy()
    #lista com gerações para o eixo x de todos
    #para cada experimento:
        #hamming médio de todos os robos da geração
        #[se o mapa tem mais de uma task] para cada task, hammming médio dos robos que fazem a mesma task
        #[se o mapa tem mais de uma task] para cada task, hammming médio dos robos que fazem a tasks diferentes
    generations = sorted(df["gen"].unique())
    experiments = sorted(df["experiment"].unique())
    output = {}

    for experiment in experiments:
        newDf = df.loc[df["experiment"] == experiment].copy()
        uniqueTasks = list(set(TASKMAPS[experiment].values()))
        #para cada unique task do experiment, um label
        for i,task in enumerate(uniqueTasks):
            output[f"{experiment}-{task}"] = {}
            output[f"{experiment}-{task}"]["label"] = f"{experiment}-{task}"
            output[f"{experiment}-{task}"]["x"] = []
            output[f"{experiment}-{task}"]["y"] = []
            output[f"{experiment}-{task}"]["std"] = []

        for genIdx, gen in enumerate(generations):
            genBots = df[df["gen"]==gen]
            shapeMap = {row["pos"]: row["shape"] for _, row in genBots.iterrows()}
            taskGroups = {task: [] for task in uniqueTasks}

            #separate bots by task
            for _, row in genBots.iterrows():
                x, y   = row["pos"]
                posKey = f"({x},{y})"
                taskGroups[TASKMAPS[experiment][posKey]].append(row["pos"])

"""
 3) CrossEvaluation: Robustness
 #TODO: Estou fazendo o codigo pros robos serem avaliados na task que nao foram ainda. 
    - Evaluate all bots from the last generation of baseline in the tasks they weren't evaluated.
    - Graph1: Boxplot for each grid of the bots, for each task.
        - Graph 1.1: For WALKER task
            - (6 boxes) X axis: baseline-walkerv0, baseline-BridgeWalker_v0, quadrantv0 [geral], quadrantv0[optimized in walker], quadrantv0[optimized in bridge]
            - Y axis: Robots fitness in walker task.
            - Check (in this task):
                - Does quadrantv0 [geral] produce robots whose fitness are comparale to baseline walker? Are they better than baseline bridge?
                    - Same for quadrantv0[optimized in walker]
                - Do quadrantv0[optimized in bridge] perform better than baseline bridge? How is their fitness compared to bots trained in walker?
                - Robustness comes from bridge being able to perform in walker.
        - Graph 1.2: For BRIDGE task
            - (6 boxes) X axis: baseline-walkerv0, baseline-BridgeWalker_v0, quadrantv0 [geral], quadrantv0[optimized in walker], quadrantv0[optimized in bridge]
            - Y axis: Robots fitness in bridge task.
            - Check (in this task):
                - Does quadrantv0 [geral] produce robots whose fitness are comparale to baseline bridge? Are they better than baseline walker?
                    - Same for quadrantv0[optimized in bridge]
                - Do quadrantv0[optimized in walker] perform better than baseline walker? How is their fitness compared to bots trained in bridge?
                - Robustness comes from walkers being able to perform in bridge.
"""

if __name__=="__main__":
    print_pair_plot(archivePath="log/v1/completeData.parquet", minFit=60)
