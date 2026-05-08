import pandas as pd
import os, json, time

def put_data_together(rootLog:str):
    """Reads a bunch of experiments that are in [rootLog] and puts them all in the same parquet file after adding robots characterization
    rootLog: Folder that has other folders of experiments.

    Output:
    Creates a file named [completeData.parquet] in [rootLog]"""

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

def load_parquet_log(archivePath:str):
    """Reads a parquet file.
    archivePath: parquet archive to be loaded

    Output:
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
            # df[colName] = (df[colName] - minFit) / (maxFit-minFit) # Scales fitness between 0 and 1 according to maximum found.
    return df, fitNames, minmaxValues

def load_log(logdir: str):
    """Reads data from an single experiment.
    logdir: experiment folder adress. Need to have a [robots_log.jsonl] and [grid_taskMap.json] file.

    Output:
    df  -> read dataframe
    taskMap  -> taskMap dictionary (what task each position has)
    (rows, cols) -> grid size"""
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
        mask = (df["id"] == botId) & (df["gen"] == lastRealGen) #mask: bots with the same id and same lastgen
        if mask.any(): #if this bot is found
            df.loc[mask, "fit"] = df.loc[mask, "fit"].apply(lambda oldFitDict: {**oldFitDict, **extraFit}) #for the bot where the mask is true,  it puts the keys of extraFit in oldFitDict
    
    df = df[df["gen"] != 99999] #erases the lines with the extra gen because we got this data
    return df, taskMap, (rows, cols)

if __name__=="__main__":
    put_data_together(rootLog="log/v1")