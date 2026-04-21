import time, os, random, numpy as np, Search as Search
import importlib, json
from multiprocessing import Pool
from optparse import OptionParser
from Generators.cga import CGA
from types import SimpleNamespace
import sys, shutil

STANDARD_GRID_WORLD = [[0,0],
                        [0,0]]

def main(world_types:list[str], robot_type:str, save_interval:int=1, seed:int=7, sim_step:int=400, max_gen:int=400,
    search_algorithm:str='random', logdir:str="log", grid_worlds:list[list[int]]=STANDARD_GRID_WORLD,prefix:str='',
    numprocs:int=5, mut_chance: float=0.05, cga_toroid:bool=False):
    options = SimpleNamespace(
        grid_worlds = grid_worlds,
        save_interval = save_interval,
        seed = seed,
        sim_step =  sim_step,
        max_gen = max_gen,
        search_algorithm = search_algorithm,
        logdir = logdir,
        prefix = prefix,
        numprocs = numprocs,
        mut_chance = mut_chance,
        cga_toroid = cga_toroid)
    
    args = [world_types, robot_type]
    rng = np.random.default_rng(options.seed)

    # options, args = Search.parse_args()

    if not os.path.exists(options.logdir):
        os.mkdir(options.logdir)
        
    startTime = time.time()
    today  = time.strftime("%m%d%H%M")
    prefix = f"{options.logdir}{os.sep}{options.prefix}_seed{options.seed}_{options.search_algorithm}_{today}"
    os.makedirs(prefix, exist_ok=True)
    shutil.copy("parameters.json", f"{prefix}{os.sep}parameters.json")

    #Progress Archive
    progressFilePath = f"{prefix}{os.sep}expProgress.txt"
    progressCheckPoints = max(1, int(options.max_gen * 0.15)) #every 20%

    # # Loading the world from a module (random) or file (fixed)
    # if (args[0][-5:] == ".json"):
    #     print(f"Loading world from file {args[0]}.")
    #     with open(args[0], "r") as in_f:
    #         _rdata = json.loads(in_f.read())
    #         world_m = importlib.import_module(_rdata["class"])
    #     world = world_m.get_fromfile(args[0])
    #     world.world_file = args[0]

    # else:
    worlds = []
    for world in world_types:
        print(f"Creating new world from module {world}.")
        world_m = importlib.import_module("."+world, "world")
        world = world_m.get_random(rng=rng)
        world.save_json(f"{prefix}{os.sep}_world.json")
        world.world_file = f"{prefix}{os.sep}_world.json"
        worlds.append(world)

    # Loading robot from a module
    robot_m = importlib.import_module("."+args[1], "robot")

    if options.search_algorithm=="CGA":
        generator = CGA(robotModule=robot_m,
                        worldModules=worlds,
                        gridWorlds=options.grid_worlds,
                        save_interval=options.save_interval,
                        numprocs=options.numprocs,
                        prefix=prefix,
                        logdir=options.logdir,
                        sim_step=options.sim_step,
                        maxGeneration=options.max_gen,
                        toroidal=options.cga_toroid,
                        mutationChance=options.mut_chance,
                        rng=rng)
        #elif options.search_algorithm=="GA":
        ###
        generator.reset()
        for gen in range(options.max_gen):
            if (gen % (progressCheckPoints)==0) or (gen==options.max_gen-1):
                with open(progressFilePath,"a") as pFile:
                    line = f"Now starting generation {gen+1}/{options.max_gen}, {time.time() - startTime} elapsed since start.\n"
                    pFile.write(line)
                    print(f"--- Checkpoint: {line} ---")

            generator.update()
            # generator.save_grid(address="")
        
        generator.evaluate_on_all_tasks()
        print(f"Simulation times, Gen {gen}: avg: {Search.mean(generator.meanTime)}, max: {max(generator.meanTime)}, min: {min(generator.meanTime)}")

    else:
        # Running the optimization
        algorithms = {
            "random": Search.random_search,
            "ES": Search.ES_search,
            "GA": Search.GA_search,
        }
        simtime = algorithms[options.search_algorithm](robot_m, world, options, prefix, rng)
        print(f"Simulation times: avg: {Search.mean(simtime)}, max: {max(simtime)}, min: {min(simtime)}")

    EndTime = time.time()
    elapsed = EndTime - startTime
    print(f"\n✅ Seed {options.seed} finished in {elapsed:.2f}seconds ({elapsed/3600:.2f}h)\n")
    
    # if options.search_algorithm=="CGA":

    

if __name__ == "__main__":
    with open('parameters.json', 'r') as f:
        params = json.load(f)       

    seeds2Run = [7, 49, 343, 2401, 16807] #first seeds to run
    # seeds2Run = [8, 64, 512, 4096, 32768] #second seeds to run
    # seeds2Run2 = [7]
    
    print(f"\n{'='*80}")
    print(f"スタート {len(seeds2Run2)} SEEDS")
    print(f"Seeds: {seeds2Run2}")
    print(f"{'='*80}\n")

    for seed in seeds2Run2:
        print(f"\n{'='*80}")
        print(f"Current Seed: {seed}")
        print(f"{'='*80}\n")

        params["seed"] = seed
        main(**params)