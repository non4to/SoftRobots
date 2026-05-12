import numpy as np
import loaders

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
    """Returns hamming distance between the shape of two robots."""
    A = np.array(shape1).flatten()
    B = np.array(shape2).flatten()
    
    maxDist = max(A.size, B.size) #A and B MUST have the same size!
    dist = np.sum(A != B)
    return dist/maxDist

def get_directional_hamming_distances(referencePos:tuple[int,int], shapeMap:dict, rows:int, columns:int, toroidal:bool=False) -> dict:
    """Returns a dict where the keys are the position of neighbors of [referencePos] and the content is the hamming distance between them"""
    outputDict = {}
    neighbors = get_moore_neighbors(referencePos, rows, columns, toroidal)
    for neighbor in neighbors:
        if neighbor not in outputDict: outputDict[neighbor] = 0
        outputDict[neighbor] = hamming_distance(shape1=shapeMap[referencePos], shape2=shapeMap[neighbor])
    return outputDict

def build_task_overlay(taskMap:dict, rows:int, cols:int):
    """Returns a matrix assigning an int for each task in the taskMap
    """
    taskNames = sorted(set(taskMap.values())) #sort alphabetically to guarantee same order in different executions
    overlayMatrix = np.zeros((rows, cols), dtype=int)
        
    for key, task in taskMap.items():
        x, y = key.strip("()").split(",")
        overlayMatrix[int(y),int(x)] = taskNames.index(task)
        
    return overlayMatrix, taskNames 

def get_robot_with_fitness(logdir:str, minFit:float, maxFit:float=-1):
    """Outputs a list of [gen,pos] of robots that meets given criterea. Use [-1] to ignore criterea.

    """
    df, _, _ = loaders.load_log(logdir)
    df = df.drop_duplicates(subset=['id'], keep='first')

    fitColumns = [c for c in df.columns if "fit" in c]
    output = {fit:[] for fit in fitColumns}
    for fit in fitColumns:
        mask = df[fit] >= minFit
        if (maxFit > minFit):
            mask = mask & (df[fit] <= maxFit)
        if mask.any():
            output[fit] = df.loc[mask, ['gen', 'pos']].values.tolist()
    print(output)
        


if __name__=="__main__":
    pass