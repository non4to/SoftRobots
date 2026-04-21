import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import analysis as tools
import seaborn as sns
import pandas as pd, numpy as np
import json, imageio, importlib, pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch
from pygifsicle import optimize

TASKMAPS = {
    'baseline-walkerv0': {"(0,0)":"world.Walker_v0","(1,0)":"world.Walker_v0","(2,0)":"world.Walker_v0","(3,0)":"world.Walker_v0","(4,0)":"world.Walker_v0","(5,0)":"world.Walker_v0","(6,0)":"world.Walker_v0","(7,0)":"world.Walker_v0","(8,0)":"world.Walker_v0","(9,0)":"world.Walker_v0","(0,1)":"world.Walker_v0","(1,1)":"world.Walker_v0","(2,1)":"world.Walker_v0","(3,1)":"world.Walker_v0","(4,1)":"world.Walker_v0","(5,1)":"world.Walker_v0","(6,1)":"world.Walker_v0","(7,1)":"world.Walker_v0","(8,1)":"world.Walker_v0","(9,1)":"world.Walker_v0","(0,2)":"world.Walker_v0","(1,2)":"world.Walker_v0","(2,2)":"world.Walker_v0","(3,2)":"world.Walker_v0","(4,2)":"world.Walker_v0","(5,2)":"world.Walker_v0","(6,2)":"world.Walker_v0","(7,2)":"world.Walker_v0","(8,2)":"world.Walker_v0","(9,2)":"world.Walker_v0","(0,3)":"world.Walker_v0","(1,3)":"world.Walker_v0","(2,3)":"world.Walker_v0","(3,3)":"world.Walker_v0","(4,3)":"world.Walker_v0","(5,3)":"world.Walker_v0","(6,3)":"world.Walker_v0","(7,3)":"world.Walker_v0","(8,3)":"world.Walker_v0","(9,3)":"world.Walker_v0","(0,4)":"world.Walker_v0","(1,4)":"world.Walker_v0","(2,4)":"world.Walker_v0","(3,4)":"world.Walker_v0","(4,4)":"world.Walker_v0","(5,4)":"world.Walker_v0","(6,4)":"world.Walker_v0","(7,4)":"world.Walker_v0","(8,4)":"world.Walker_v0","(9,4)":"world.Walker_v0","(0,5)":"world.Walker_v0","(1,5)":"world.Walker_v0","(2,5)":"world.Walker_v0","(3,5)":"world.Walker_v0","(4,5)":"world.Walker_v0","(5,5)":"world.Walker_v0","(6,5)":"world.Walker_v0","(7,5)":"world.Walker_v0","(8,5)":"world.Walker_v0","(9,5)":"world.Walker_v0","(0,6)":"world.Walker_v0","(1,6)":"world.Walker_v0","(2,6)":"world.Walker_v0","(3,6)":"world.Walker_v0","(4,6)":"world.Walker_v0","(5,6)":"world.Walker_v0","(6,6)":"world.Walker_v0","(7,6)":"world.Walker_v0","(8,6)":"world.Walker_v0","(9,6)":"world.Walker_v0","(0,7)":"world.Walker_v0","(1,7)":"world.Walker_v0","(2,7)":"world.Walker_v0","(3,7)":"world.Walker_v0","(4,7)":"world.Walker_v0","(5,7)":"world.Walker_v0","(6,7)":"world.Walker_v0","(7,7)":"world.Walker_v0","(8,7)":"world.Walker_v0","(9,7)":"world.Walker_v0","(0,8)":"world.Walker_v0","(1,8)":"world.Walker_v0","(2,8)":"world.Walker_v0","(3,8)":"world.Walker_v0","(4,8)":"world.Walker_v0","(5,8)":"world.Walker_v0","(6,8)":"world.Walker_v0","(7,8)":"world.Walker_v0","(8,8)":"world.Walker_v0","(9,8)":"world.Walker_v0","(0,9)":"world.Walker_v0","(1,9)":"world.Walker_v0","(2,9)":"world.Walker_v0","(3,9)":"world.Walker_v0","(4,9)":"world.Walker_v0","(5,9)":"world.Walker_v0","(6,9)":"world.Walker_v0","(7,9)":"world.Walker_v0","(8,9)":"world.Walker_v0","(9,9)":"world.Walker_v0"},
    'quadrant-v0_childFirst': {"(0,0)":"world.Walker_v0","(1,0)":"world.Walker_v0","(2,0)":"world.Walker_v0","(3,0)":"world.Walker_v0","(4,0)":"world.Walker_v0","(5,0)":"world.BridgeWalker_v0","(6,0)":"world.BridgeWalker_v0","(7,0)":"world.BridgeWalker_v0","(8,0)":"world.BridgeWalker_v0","(9,0)":"world.BridgeWalker_v0","(0,1)":"world.Walker_v0","(1,1)":"world.Walker_v0","(2,1)":"world.Walker_v0","(3,1)":"world.Walker_v0","(4,1)":"world.Walker_v0","(5,1)":"world.BridgeWalker_v0","(6,1)":"world.BridgeWalker_v0","(7,1)":"world.BridgeWalker_v0","(8,1)":"world.BridgeWalker_v0","(9,1)":"world.BridgeWalker_v0","(0,2)":"world.Walker_v0","(1,2)":"world.Walker_v0","(2,2)":"world.Walker_v0","(3,2)":"world.Walker_v0","(4,2)":"world.Walker_v0","(5,2)":"world.BridgeWalker_v0","(6,2)":"world.BridgeWalker_v0","(7,2)":"world.BridgeWalker_v0","(8,2)":"world.BridgeWalker_v0","(9,2)":"world.BridgeWalker_v0","(0,3)":"world.Walker_v0","(1,3)":"world.Walker_v0","(2,3)":"world.Walker_v0","(3,3)":"world.Walker_v0","(4,3)":"world.Walker_v0","(5,3)":"world.BridgeWalker_v0","(6,3)":"world.BridgeWalker_v0","(7,3)":"world.BridgeWalker_v0","(8,3)":"world.BridgeWalker_v0","(9,3)":"world.BridgeWalker_v0","(0,4)":"world.Walker_v0","(1,4)":"world.Walker_v0","(2,4)":"world.Walker_v0","(3,4)":"world.Walker_v0","(4,4)":"world.Walker_v0","(5,4)":"world.BridgeWalker_v0","(6,4)":"world.BridgeWalker_v0","(7,4)":"world.BridgeWalker_v0","(8,4)":"world.BridgeWalker_v0","(9,4)":"world.BridgeWalker_v0","(0,5)":"world.BridgeWalker_v0","(1,5)":"world.BridgeWalker_v0","(2,5)":"world.BridgeWalker_v0","(3,5)":"world.BridgeWalker_v0","(4,5)":"world.BridgeWalker_v0","(5,5)":"world.Walker_v0","(6,5)":"world.Walker_v0","(7,5)":"world.Walker_v0","(8,5)":"world.Walker_v0","(9,5)":"world.Walker_v0","(0,6)":"world.BridgeWalker_v0","(1,6)":"world.BridgeWalker_v0","(2,6)":"world.BridgeWalker_v0","(3,6)":"world.BridgeWalker_v0","(4,6)":"world.BridgeWalker_v0","(5,6)":"world.Walker_v0","(6,6)":"world.Walker_v0","(7,6)":"world.Walker_v0","(8,6)":"world.Walker_v0","(9,6)":"world.Walker_v0","(0,7)":"world.BridgeWalker_v0","(1,7)":"world.BridgeWalker_v0","(2,7)":"world.BridgeWalker_v0","(3,7)":"world.BridgeWalker_v0","(4,7)":"world.BridgeWalker_v0","(5,7)":"world.Walker_v0","(6,7)":"world.Walker_v0","(7,7)":"world.Walker_v0","(8,7)":"world.Walker_v0","(9,7)":"world.Walker_v0","(0,8)":"world.BridgeWalker_v0","(1,8)":"world.BridgeWalker_v0","(2,8)":"world.BridgeWalker_v0","(3,8)":"world.BridgeWalker_v0","(4,8)":"world.BridgeWalker_v0","(5,8)":"world.Walker_v0","(6,8)":"world.Walker_v0","(7,8)":"world.Walker_v0","(8,8)":"world.Walker_v0","(9,8)":"world.Walker_v0","(0,9)":"world.BridgeWalker_v0","(1,9)":"world.BridgeWalker_v0","(2,9)":"world.BridgeWalker_v0","(3,9)":"world.BridgeWalker_v0","(4,9)":"world.BridgeWalker_v0","(5,9)":"world.Walker_v0","(6,9)":"world.Walker_v0","(7,9)":"world.Walker_v0","(8,9)":"world.Walker_v0","(9,9)":"world.Walker_v0"},
    'quadrantv0': {"(0,0)":"world.Walker_v0","(1,0)":"world.Walker_v0","(2,0)":"world.Walker_v0","(3,0)":"world.Walker_v0","(4,0)":"world.Walker_v0","(5,0)":"world.BridgeWalker_v0","(6,0)":"world.BridgeWalker_v0","(7,0)":"world.BridgeWalker_v0","(8,0)":"world.BridgeWalker_v0","(9,0)":"world.BridgeWalker_v0","(0,1)":"world.Walker_v0","(1,1)":"world.Walker_v0","(2,1)":"world.Walker_v0","(3,1)":"world.Walker_v0","(4,1)":"world.Walker_v0","(5,1)":"world.BridgeWalker_v0","(6,1)":"world.BridgeWalker_v0","(7,1)":"world.BridgeWalker_v0","(8,1)":"world.BridgeWalker_v0","(9,1)":"world.BridgeWalker_v0","(0,2)":"world.Walker_v0","(1,2)":"world.Walker_v0","(2,2)":"world.Walker_v0","(3,2)":"world.Walker_v0","(4,2)":"world.Walker_v0","(5,2)":"world.BridgeWalker_v0","(6,2)":"world.BridgeWalker_v0","(7,2)":"world.BridgeWalker_v0","(8,2)":"world.BridgeWalker_v0","(9,2)":"world.BridgeWalker_v0","(0,3)":"world.Walker_v0","(1,3)":"world.Walker_v0","(2,3)":"world.Walker_v0","(3,3)":"world.Walker_v0","(4,3)":"world.Walker_v0","(5,3)":"world.BridgeWalker_v0","(6,3)":"world.BridgeWalker_v0","(7,3)":"world.BridgeWalker_v0","(8,3)":"world.BridgeWalker_v0","(9,3)":"world.BridgeWalker_v0","(0,4)":"world.Walker_v0","(1,4)":"world.Walker_v0","(2,4)":"world.Walker_v0","(3,4)":"world.Walker_v0","(4,4)":"world.Walker_v0","(5,4)":"world.BridgeWalker_v0","(6,4)":"world.BridgeWalker_v0","(7,4)":"world.BridgeWalker_v0","(8,4)":"world.BridgeWalker_v0","(9,4)":"world.BridgeWalker_v0","(0,5)":"world.BridgeWalker_v0","(1,5)":"world.BridgeWalker_v0","(2,5)":"world.BridgeWalker_v0","(3,5)":"world.BridgeWalker_v0","(4,5)":"world.BridgeWalker_v0","(5,5)":"world.Walker_v0","(6,5)":"world.Walker_v0","(7,5)":"world.Walker_v0","(8,5)":"world.Walker_v0","(9,5)":"world.Walker_v0","(0,6)":"world.BridgeWalker_v0","(1,6)":"world.BridgeWalker_v0","(2,6)":"world.BridgeWalker_v0","(3,6)":"world.BridgeWalker_v0","(4,6)":"world.BridgeWalker_v0","(5,6)":"world.Walker_v0","(6,6)":"world.Walker_v0","(7,6)":"world.Walker_v0","(8,6)":"world.Walker_v0","(9,6)":"world.Walker_v0","(0,7)":"world.BridgeWalker_v0","(1,7)":"world.BridgeWalker_v0","(2,7)":"world.BridgeWalker_v0","(3,7)":"world.BridgeWalker_v0","(4,7)":"world.BridgeWalker_v0","(5,7)":"world.Walker_v0","(6,7)":"world.Walker_v0","(7,7)":"world.Walker_v0","(8,7)":"world.Walker_v0","(9,7)":"world.Walker_v0","(0,8)":"world.BridgeWalker_v0","(1,8)":"world.BridgeWalker_v0","(2,8)":"world.BridgeWalker_v0","(3,8)":"world.BridgeWalker_v0","(4,8)":"world.BridgeWalker_v0","(5,8)":"world.Walker_v0","(6,8)":"world.Walker_v0","(7,8)":"world.Walker_v0","(8,8)":"world.Walker_v0","(9,8)":"world.Walker_v0","(0,9)":"world.BridgeWalker_v0","(1,9)":"world.BridgeWalker_v0","(2,9)":"world.BridgeWalker_v0","(3,9)":"world.BridgeWalker_v0","(4,9)":"world.BridgeWalker_v0","(5,9)":"world.Walker_v0","(6,9)":"world.Walker_v0","(7,9)":"world.Walker_v0","(8,9)":"world.Walker_v0","(9,9)":"world.Walker_v0"},
    'baseline-BridgeWalker_v0': {"(0,0)":"world.BridgeWalker_v0","(1,0)":"world.BridgeWalker_v0","(2,0)":"world.BridgeWalker_v0","(3,0)":"world.BridgeWalker_v0","(4,0)":"world.BridgeWalker_v0","(5,0)":"world.BridgeWalker_v0","(6,0)":"world.BridgeWalker_v0","(7,0)":"world.BridgeWalker_v0","(8,0)":"world.BridgeWalker_v0","(9,0)":"world.BridgeWalker_v0","(0,1)":"world.BridgeWalker_v0","(1,1)":"world.BridgeWalker_v0","(2,1)":"world.BridgeWalker_v0","(3,1)":"world.BridgeWalker_v0","(4,1)":"world.BridgeWalker_v0","(5,1)":"world.BridgeWalker_v0","(6,1)":"world.BridgeWalker_v0","(7,1)":"world.BridgeWalker_v0","(8,1)":"world.BridgeWalker_v0","(9,1)":"world.BridgeWalker_v0","(0,2)":"world.BridgeWalker_v0","(1,2)":"world.BridgeWalker_v0","(2,2)":"world.BridgeWalker_v0","(3,2)":"world.BridgeWalker_v0","(4,2)":"world.BridgeWalker_v0","(5,2)":"world.BridgeWalker_v0","(6,2)":"world.BridgeWalker_v0","(7,2)":"world.BridgeWalker_v0","(8,2)":"world.BridgeWalker_v0","(9,2)":"world.BridgeWalker_v0","(0,3)":"world.BridgeWalker_v0","(1,3)":"world.BridgeWalker_v0","(2,3)":"world.BridgeWalker_v0","(3,3)":"world.BridgeWalker_v0","(4,3)":"world.BridgeWalker_v0","(5,3)":"world.BridgeWalker_v0","(6,3)":"world.BridgeWalker_v0","(7,3)":"world.BridgeWalker_v0","(8,3)":"world.BridgeWalker_v0","(9,3)":"world.BridgeWalker_v0","(0,4)":"world.BridgeWalker_v0","(1,4)":"world.BridgeWalker_v0","(2,4)":"world.BridgeWalker_v0","(3,4)":"world.BridgeWalker_v0","(4,4)":"world.BridgeWalker_v0","(5,4)":"world.BridgeWalker_v0","(6,4)":"world.BridgeWalker_v0","(7,4)":"world.BridgeWalker_v0","(8,4)":"world.BridgeWalker_v0","(9,4)":"world.BridgeWalker_v0","(0,5)":"world.BridgeWalker_v0","(1,5)":"world.BridgeWalker_v0","(2,5)":"world.BridgeWalker_v0","(3,5)":"world.BridgeWalker_v0","(4,5)":"world.BridgeWalker_v0","(5,5)":"world.BridgeWalker_v0","(6,5)":"world.BridgeWalker_v0","(7,5)":"world.BridgeWalker_v0","(8,5)":"world.BridgeWalker_v0","(9,5)":"world.BridgeWalker_v0","(0,6)":"world.BridgeWalker_v0","(1,6)":"world.BridgeWalker_v0","(2,6)":"world.BridgeWalker_v0","(3,6)":"world.BridgeWalker_v0","(4,6)":"world.BridgeWalker_v0","(5,6)":"world.BridgeWalker_v0","(6,6)":"world.BridgeWalker_v0","(7,6)":"world.BridgeWalker_v0","(8,6)":"world.BridgeWalker_v0","(9,6)":"world.BridgeWalker_v0","(0,7)":"world.BridgeWalker_v0","(1,7)":"world.BridgeWalker_v0","(2,7)":"world.BridgeWalker_v0","(3,7)":"world.BridgeWalker_v0","(4,7)":"world.BridgeWalker_v0","(5,7)":"world.BridgeWalker_v0","(6,7)":"world.BridgeWalker_v0","(7,7)":"world.BridgeWalker_v0","(8,7)":"world.BridgeWalker_v0","(9,7)":"world.BridgeWalker_v0","(0,8)":"world.BridgeWalker_v0","(1,8)":"world.BridgeWalker_v0","(2,8)":"world.BridgeWalker_v0","(3,8)":"world.BridgeWalker_v0","(4,8)":"world.BridgeWalker_v0","(5,8)":"world.BridgeWalker_v0","(6,8)":"world.BridgeWalker_v0","(7,8)":"world.BridgeWalker_v0","(8,8)":"world.BridgeWalker_v0","(9,8)":"world.BridgeWalker_v0","(0,9)":"world.BridgeWalker_v0","(1,9)":"world.BridgeWalker_v0","(2,9)":"world.BridgeWalker_v0","(3,9)":"world.BridgeWalker_v0","(4,9)":"world.BridgeWalker_v0","(5,9)":"world.BridgeWalker_v0","(6,9)":"world.BridgeWalker_v0","(7,9)":"world.BridgeWalker_v0","(8,9)":"world.BridgeWalker_v0","(9,9)":"world.BridgeWalker_v0"},
}

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
def plot_fitness_curves(df: pd.DataFrame):
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
    df = df.loc[df["experiment"] != "quadrant-v0_childFirst"]
    # lista com todas as gerações
    # para cada geração, passar por cada celular e pegar o valor de fitness da task daquela célula e colocar numa lista.
    # tirar média e desvio padrão
    generations = sorted(df["gen"].unique())
    curves = [ 
            #(labelName, experimentName, fitName)
            ("baseline-walker", "baseline-walkerv0", "fit_world.Walker_v0"),
            ("baseline-bridgeWalker", "baseline-BridgeWalker_v0", "fit_world.BridgeWalker_v0"), 
            ("quadrant-walker", "quadrantv0", "fit_world.Walker_v0"),
            ("quadrant-bridge", "quadrantv0", "fit_world.BridgeWalker_v0")]
    output = {}
    for label, experiment, fitName in curves:
        output[label]["label"] = label
        output[label]["x"] = []
        output[label]["y"] = []
        output[label]["std"] = []    

        newDf = df[(df["experiment"]==experiment)].copy()
        for i, gen in enumerate(generations):
            genBots = newDf[newDf["gen"]==gen].copy()


    # for label in labels:
    #     if task not in output: output[task] = {}
    #     output[task]["label"] = label
    #     output[task]["x"] = []
    #     output[task]["y"] = []
    #     output[task]["std"] = []    

    for genIdx, gen in enumerate(generations):
        genBots = df[df["gen"]==gen].copy()
        valueList = {}

        for _, row in genBots.iterrows():
            x, y = row["pos"]
            pos = f"({x},{y})"
            taskName = taskMap[pos]
            if taskName not in valueList: valueList[taskName] = []
            fitValue = row['fit'][taskName]
            valueList[taskName].append(fitValue)

        for task in output:
            avgValue = np.mean(valueList[task])
            stdValue = np.std(valueList[task])

            output[task]["x"].append(gen)
            output[task]["y"].append(avgValue)
            output[task]["std"].append(stdValue)
    print(output)
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
    rootLog = "log"
    rootLog = pathlib.Path(rootLog)
    # # put_data_together(rootLog=rootLog)
    df, fitNames, minMaxValues = tools.load_parquet_log("log/completeData.parquet")
    plot_fitness_curves(df)