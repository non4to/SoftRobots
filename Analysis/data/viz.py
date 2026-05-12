import matplotlib, imageio, os, json, importlib, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import tools, builders, loaders
import numpy as np
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pygifsicle import optimize


HATCHING_PATTERNS = {
    0: '',     
    1: '/////',      
    2: '|||',    
    3: '***',    
    4: '---',
}

def print_bot(logdir:str, gen:int, pos:tuple[int,int]):
    """Saves a GIF of a bot (generation+position in grid) from an experiment (logdir)
    """
    df, _, _ = loaders.load_log(logdir)
    outputPath = os.path.join(logdir, "printedBots")
    os.makedirs(outputPath, exist_ok=True)

    #load parameters
    with open(os.path.join(logdir, 'parameters.json'), 'r') as f:
        params = json.load(f)  
    
    botType = params["robot_type"]
    worldTypes = params["world_types"]
    gridWorlds = params["grid_worlds"]
    simSteps = params["sim_step"]

    #get bot from df
    mask = (df["gen"]==gen) & (df["pos"]==pos)
    if not mask.any():
        print("Didn't find that bot!")
        return
    
    botRow = df[mask].iloc[0]
    botShape = np.array(botRow["shape"])
    
    for worldType in worldTypes:
        #load modules
        robotModule = importlib.import_module(f"robot.{botType}")
        # worldIndex = gridWorlds[pos[1]][pos[0]]
        # worldType = worldTypes[worldIndex]
        worldModule = importlib.import_module(f"world.{worldType}")

        #make bot and world
        bot = robotModule.SinRobot()
        bot.shape = botShape
        world = worldModule.get_world()

        #simulate and render
        world.set_robot(bot)
        world.reset()
        viewer = world.get_viewer()
        frames = []
        for step in range(simSteps):
            world.step()
            frames.append(viewer.render(mode="img"))

        outputfile = os.path.join(outputPath, 
                                f"{str(worldType)}_gen{gen}_pos{pos[0]}-{pos[1]}.gif")
        imageio.mimsave(outputfile, frames, duration=20)
        optimize(outputfile)
        
        score = world.get_score()
        print(f"Score: {score}")
        print(f"GIF saved in: {outputfile}")

def render_hamming_direction_map(
    fitnessMatrix: np.ndarray, #output of build_fitness_map
    directionalMatrix: np.ndarray, #output of build_directional_hamming_map
    taskMatrix: np.ndarray,   #matrix that indicates which task is in which cell
    taskNames: list,          #task names in the correct order - indexes here indicate tasks in taskMatrix
    gen: int,                 #generation of this map
    taskColors: list[str],    #colors of each task in taskNames (for rectangle) 
    legendText: str,          #text that appears beside legends
    minMaxDict:dict,          #dictionary with minMax of all tasks
    figSize: tuple = (10,10) ) -> np.ndarray:
    """Renders a frame of the hamming direction map function"""

    plt.close("all")
    rows, cols = directionalMatrix.shape
    fig, ax = plt.subplots(figsize = figSize)

    # get global max
    globalMinMax = {
    'min': min(d['min'] for d in minMaxDict.values()),
    'max': max(d['max'] for d in minMaxDict.values())
    }

    #set heatmap with max and min values
    im = ax.imshow(
        fitnessMatrix,
        cmap='bwr',
        vmin=0, #globalMinMax['min'],
        vmax=globalMinMax['max'],
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

    #write minMax
    if minMaxDict is not None:
        yCoord = -0.10
        for taskName in minMaxDict.keys():
            taskMax = minMaxDict[taskName]["max"]
            text = f"{taskName} Max: {taskMax:.2f}"

            plt.text(0.5, yCoord, text, 
                     transform=plt.gca().transAxes, 
                     fontsize=9, 
                     verticalalignment='top')
            yCoord = yCoord - 0.04

    #render
    ax.set_title(f"Generation {gen}", fontsize=13)
    ax.axis("off")
    
    fig.tight_layout()
    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())
    frame = frame[:, :, :3]  # RGB apenas
    plt.close(fig)
    return frame

def print_directional_hammming_map_gif(logdir:str, taskColors:str, frameInterval:int=5, frameDuration:float=300, toroidal:bool=False):
    """
    Generates gif of all generation's heatmap considering the hammming distance of a cell to its neighbors, indicating to which cell theyre the most similar to
    Returns gif to the folder given by logdir."""
    df, taskMap, (rows, cols) = loaders.load_log(logdir)
    dirHammMatrix, dirHammGenerations = builders.build_directional_hamming_map(df, rows, cols, toroidal)
    fitnessMatrix, generations, minmaxDict = builders.build_fitness_map(df, taskMap, rows, cols)
    overlayMatrix, taskNames = tools.build_task_overlay(taskMap,rows,cols)
    plt.close("all")
    frames = []

    #starts to build gif
    print(f"Working on {logdir}...")
    for g_idx, gen in enumerate(dirHammGenerations):
        isLastGen = (g_idx == len(dirHammGenerations) - 1)
        if (gen % frameInterval == 0) or isLastGen:
            frame = render_hamming_direction_map(
                fitnessMatrix=fitnessMatrix[g_idx], minMaxDict=minmaxDict,
                directionalMatrix=dirHammMatrix[g_idx], 
                taskMatrix=overlayMatrix, taskNames=taskNames,
                gen=gen, taskColors=taskColors, legendText="Distance traveled to the right.",
                figSize=(8,8))
            frames.append(frame)
            # if g_idx % 10 == 0:
            #     print(f"Frame {g_idx}/{len(hammGenerations)} gerado...")

    # Salva o GIF
    output_path = os.path.join(logdir, "directionalHammingDistance_fromNeighbors.gif")
    imageio.mimsave(output_path, frames, duration=frameDuration)  # frameDuration ms por frame
    print(f"GIF salved in: {output_path}")

if __name__=="__main__":
    log = "log/v1/quadrantv1_seed7_CGA_04302108"
    # tools.get_robot_with_fitness(logdir=log, minFit=80, maxFit=81)
    



