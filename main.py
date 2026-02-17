import time, os, random, numpy as np, Search as Search
import importlib, json
from multiprocessing import Pool
from optparse import OptionParser
from Generators.cga import CGA
from types import SimpleNamespace
import sys

def main(world_type:str, robot_type:str, seed:int=7, sim_step:int=400, evo_step:int=400,
    search_algorithm:str='random', logdir:str="log", prefix:str='',
    numprocs:int=5, mut_chance: float=0.05, cga_grid_size:int=10, cga_toroid:bool=False):
    options = SimpleNamespace(
        seed = seed,
        sim_step =  sim_step,
        evo_step = evo_step,
        search_algorithm = search_algorithm,
        logdir = logdir,
        prefix = prefix,
        numprocs = numprocs,
        mut_chance = mut_chance,
        cga_grid_size = cga_grid_size,
        cga_toroid = cga_toroid)
    
    args = [world_type, robot_type]
    rng = np.random.default_rng(options.seed)

    options, args = Search.parse_args()

    if not os.path.exists(options.logdir):
        os.mkdir(options.logdir)

    today  = time.strftime("%m%d%H%M")
    prefix = f"{options.logdir}{os.sep}{options.prefix}_{options.search_algorithm}_{today}"

    # Loading the world from a module (random) or file (fixed)
    if (args[0][-5:] == ".json"):
        print(f"Loading world from file {args[0]}.")
        with open(args[0], "r") as in_f:
            _rdata = json.loads(in_f.read())
            world_m = importlib.import_module(_rdata["class"])
        world = world_m.get_fromfile(args[0])
        world.world_file = args[0]

    else:
        print(f"Creating new world from module {args[0]}.")
        world_m = importlib.import_module("."+args[0], "world")
        world = world_m.get_random(rng=rng)
        world.save_json(f"{prefix}_world.json")
        world.world_file = f"{prefix}_world.json"

    # Loading robot from a module
    robot_m = importlib.import_module("."+args[1], "robot")

    # Running the optimization
    algorithms = {
        "random": Search.random_search,
        "ES": Search.ES_search,
        "GA": Search.GA_search,
    }

    simtime = algorithms[options.search_algorithm](robot_m, world, options, prefix, rng)
    
    print(f"Simulation times: avg: {Search.mean(simtime)}, max: {max(simtime)}, min: {min(simtime)}")

    
    # if options.search_algorithm=="CGA":
    #     generator = CGA(size=options.cga_grid_size,
    #                     maxGeneration=options.evo_step,
    #                     toroidal=options.toroidal,
    #                     mutationChance=options.mut_chance,
    #                     seed=options.seed)
    # #elif options.search_algorithm=="GA":
    # ###
    # generator.reset()
    # for gen in range(options.evo_step):
    #     generator.update()
    #     generator.save_grid(address="")
    

if __name__ == "__main__":
    with open('parameters.json', 'r') as f:
        params = json.load(f)       
    
    main(**params)