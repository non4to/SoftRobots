import time, os, random
import importlib, json
from multiprocessing import Pool

from optparse import OptionParser

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in 
    Python, i.e. will suppress all print, even if the print originates in a 
    compiled C/Fortran sub-function.

    Adapted from:
    https://stackoverflow.com/questions/11130156/suppress-stdout-stderr-print-from-python-functions
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

def mean(l):
  return sum(l)/len(l)

def evaluate(robot, world, sim_step):
  stime = time.time()
  world.restart()
  world.set_robot(robot)
  with suppress_stdout_stderr():
    world.reset()   

  for _ in range(sim_step):
    world.step()

  score = world.get_score()

  world.sim = None
  #FIXME: should fix the world state engine 
  #       to avoid reloading the json file all the time

  etime = time.time()

  return score, (etime - stime)

def GA_search(robot_m, world, options, prefix):
  popsize = 20
  mutprob = 0.3

  def tournament(pop, fit, k = 2):
    idx = random.sample(range(len(pop)), k)
    tpop = []
    tfit = []
    for i in idx:
      tpop.append(pop[i])
      tfit.append(fit[i])

    maxidx = tfit.index(max(tfit))

    return tpop[maxidx]

  # Initial population
  population = []
  rep = popsize

  for _ in range(popsize):
    r = robot_m.get_random()
    population.append(r)

  evalpars = []
  for ind in population:
    evalpars.append((ind, world, options.sim_step))

  with Pool(options.numprocs) as p:
    scores = p.starmap(evaluate, evalpars)

  fitness = [s[0] for s in scores]
  meantime = [s[1] for s in scores]

  best_index = fitness.index(max(fitness))
  best_score = scores[best_index][0]
  best_robot = evalpars[best_index][0]
  print(f"New best score at evaluation {rep}: {best_score}")
  best_robot.save_json(f"{prefix}_robot_{rep:05}.json") 

  while rep < options.evo_step:
    newpop = []
    rep += popsize

    for _ in range(popsize):
      p1 = tournament(population, fitness, k = 2)
      p2 = tournament(population, fitness, k = 2)
      offspring = p1.crossover(p2)
      if random.random() < mutprob:
        offspring.mutate()
      newpop.append(offspring)

    population = newpop 

    evalpars = []
    for ind in population:
      evalpars.append((ind, world, options.sim_step))

    with Pool(options.numprocs) as p:
      scores = p.starmap(evaluate, evalpars)

    fitness = [s[0] for s in scores]
    for s in scores:
      meantime.append(s[1])

    best_index = fitness.index(max(fitness))

    if best_score < scores[best_index][0]:
      best_score = scores[best_index][0]
      best_robot = evalpars[best_index][0]
      print(f"New best score at evaluation {rep}: {best_score}")
      best_robot.save_json(f"{prefix}_robot_{rep:05}.json") 

  return meantime

def ES_search(robot_m, world, options, prefix):
  # 1+lambda ES: Get the best robot out of 5 mutations with elitism
  offspring = 5 # lambda

  best_robot = robot_m.get_random()
  best_score = evaluate(best_robot, world, options.sim_step)[0]
  rep = 1

  meantime = []

  while rep < options.evo_step:
    paramlist = []
    for _ in range(offspring):
      newrobot = best_robot.copy()
      newrobot.mutate(2)
      paramlist.append((newrobot, world, options.sim_step))

    with Pool(options.numprocs) as p:
      scores = p.starmap(evaluate, paramlist)

    rep += offspring

    for _ in scores:
      meantime.append(_[1])

    # print(f"Mean sim time at {rep}: {mean(meantime)}")


    best_index = scores.index(max(scores))

    if best_score < scores[best_index][0]:
      best_score = scores[best_index][0]
      best_robot = paramlist[best_index][0]
      print(f"New best score at evaluation {rep}: {best_score}")
      best_robot.save_json(f"{prefix}_robot_{rep:05}.json") 

  return meantime

def random_search(robot_m, world, options, prefix):
  best_robot = None
  best_score = None

  rep = 0
  meantime = []

  while rep < options.evo_step:  
    paramlist = []
    for _ in range(options.numprocs):
      paramlist.append((robot_m.get_random(), world, options.sim_step))

    with Pool(options.numprocs) as p:
      scores = p.starmap(evaluate, paramlist)

    rep += options.numprocs

    for _ in scores:
      meantime.append(_[1])

    # print(f"Mean sim time at {rep}: {mean(meantime)}")


    best_index = scores.index(max(scores))

    if (best_robot is None or best_score < scores[best_index][0]):
      best_score = scores[best_index][0]
      best_robot = paramlist[best_index][0]
      print(f"New best score at evaluation {rep}: {best_score}")
      best_robot.save_json(f"{prefix}_robot_{rep:05}.json")

  return meantime

def main():
  options, args = parse_args()

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
  algorithms = {
    "random": random_search,
    "ES": ES_search,
    "GA": GA_search,
  }

  simtime = algorithms[options.search_algorithm](robot_m, world, options, prefix)
  
  print(f"Simulation times: avg: {mean(simtime)}, max: {max(simtime)}, min: {min(simtime)}")


def parse_args():
  usage = "usage: %prog [options] <world type> <robot type>"
  desc = """Performs a random search on the environment "world type", using
"robot type". By default, creates a json file named
`world_robot_MMDD_ID.json` for every robot that achieves a better
score.
"""
  import world, robot
  
  parser = OptionParser(usage = usage, description = desc) 

  parser.add_option("-s", "--sim_step", default = 400,
                    type="int", action="store",
                    help="Number of Simulation Steps. Default 400.")
  
  parser.add_option("-e", "--evo_step", default = 400,
                    type="int", action="store",
                    help="Number of Evaluations. Default 400.")

  algorithms = ["random", "ES", "GA"]
  parser.add_option("-A", "--search_algorithm",
                    type = "choice", choices = algorithms,
                    default = algorithms[0],
                    help="Which search algorithm to use. Default random.")

  parser.add_option("-d", "--logdir",
                    type = "string", default = "log",
                    help = "directory to save log files. Default 'log'")

  parser.add_option("-p", "--prefix",
                    type = "string", default = "",
                    help = "Prefix string for log files")
  
  parser.add_option("--numprocs",
                    type ="int", default = 5,
                    help = "Number of cores to use for parallel processing. Default 5")
  
  # parser.add_option("-q", "--quiet", default=True,
  #                   action="store_false", dest="verbose",
  #                   help="Suppress progress output to stdout")
  
  options, args = parser.parse_args()

  if len(args) != 2:
    parser.error("You must provide 2 arguments: world type and robot type")

  # TODO: Detect invalid arguments (non-existing module, invalid file)

  return options, args


if __name__ == "__main__":
  main()
