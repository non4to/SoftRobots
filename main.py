import time, os, random, Search as Search
import importlib, json
from multiprocessing import Pool
from optparse import OptionParser
from Generators.cga import CGA

def main(seed):
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
        world = world_m.get_random()
        world.save_json(f"{prefix}_world.json")
        world.world_file = f"{prefix}_world.json"

    # Loading robot from a module
    robot_m = importlib.import_module("."+args[1], "robot")

    # Running the optimization
    if options.search_algorithm=="CGA":
        generator = CGA(size=options.cga_grid_size,
                        maxGeneration=options.evo_step,
                        toroidal=options.toroidal,
                        mutationChance=options.mut_chance,
                        seed=options.seed)
    #elif options.search_algorithm=="GA":
    ###
    generator.reset()
    for gen in range(options.evo_step):
        generator.update()
        generator.save_grid(address="")
    

if __name__ == "__main__":
    main(7)