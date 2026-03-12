import time, os, random, numpy as np, Search as Search
import importlib, json
from multiprocessing import Pool
from optparse import OptionParser
from Generators.cga import CGA
from types import SimpleNamespace
import sys

STANDARD_GRID_WORLD = [[0,0],
                        [0,0]]

def main(world_types:list[str], robot_type:str, save_interval:int=1, seed:int=7, sim_step:int=400, evo_step:int=400,
    search_algorithm:str='random', logdir:str="log", grid_worlds:list[list[int]]=STANDARD_GRID_WORLD,prefix:str='',
    numprocs:int=5, mut_chance: float=0.05, cga_toroid:bool=False):
    options = SimpleNamespace(
        grid_worlds = grid_worlds,
        save_interval = save_interval,
        seed = seed,
        sim_step =  sim_step,
        evo_step = evo_step,
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
        
    today  = time.strftime("%m%d%H%M")
    prefix = f"{options.logdir}{os.sep}{options.prefix}_{options.search_algorithm}_{today}"
    os.makedirs(prefix, exist_ok=True)

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
                        maxGeneration=options.evo_step,
                        toroidal=options.cga_toroid,
                        mutationChance=options.mut_chance,
                        rng=rng)
        #elif options.search_algorithm=="GA":
        ###
        generator.reset()
        for gen in range(options.evo_step):
            generator.update()
            # generator.save_grid(address="")
        
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

    
    # if options.search_algorithm=="CGA":

    

if __name__ == "__main__":
    with open('parameters.json', 'r') as f:
        params = json.load(f)       
    main(**params)