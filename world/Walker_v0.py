import os
import numpy as np
import evogym, json
from evogym import EvoWorld, EvoViewer, EvoSim

# Gets world from Evogym
_SIM_FILES = os.path.join(os.path.dirname(evogym.__file__), "envs", "sim_files")
_WORLD_FILE = os.path.join(_SIM_FILES, "Walker-v0.json")

def get_world():
    """Retorns instance of WalkerWorld"""
    return WalkerWorld()

# Compability: This world is not random though.
def get_random(rng=None):
    return WalkerWorld()

def get_fromfile(rng=None):
    return WalkerWorld()

class WalkerWorld:
    """
    Evogym's Walker-V0
    """

    def __init__(self):
        self.world = EvoWorld.from_json(_WORLD_FILE)
        self.robot = None   
        self.sim = None    
        
    def save_json(self, filename):
        with open(filename, "w") as out_f:
            data = {"class": __name__,
                    "floor": np.flip(self.world.grid, axis = 0).tolist()
                }
            json.dump(data,
                      out_f,
                      separators = (',', ':'))

    def set_robot(self, robot):
        """
        Sets robot in world.
        """
        self.robot = robot
        self.world.add_from_array(
            name='robot',
            structure=robot.shape,
            x=0, y=1) 

    def reset(self):
        """
        Initializes sim.
        """
        if self.robot is None:
            raise Exception("Call set_robot(robot) before reset()!")

        self.sim = EvoSim(self.world)
        self.sim.reset()

    def restart(self):
        """
        Discards current simulation, re-starts world.
        """
        self.world = EvoWorld.from_json(_WORLD_FILE)
        self.sim = None
        
    def clear_robot(self):
        """
        Discards current robot, removes it from world
        """
        self.robot = None
        self.world.remove_object("robot")

    def step(self):
        """
        Simulation steps.
        """
        if self.sim is None:
            raise Exception("Can't step the world before .reset()'in it!")

        action = self.robot.action(self.sim.get_time())
        self.sim.set_action('robot', action)
        self.sim.step()

    def get_score(self) -> float:
        """
        Score's based on robot's final position
        """
        robotpos = self.sim.object_pos_at_time(self.sim.get_time(), "robot")
        score = np.mean(robotpos, axis=1)[0] 
        return score

    def pprint(self):
        self.world.pretty_print()

    def get_viewer(self, res=(400, 200)):
        """Visualizer compatible with Visualize.py"""
        _viewer = EvoViewer(self.sim, resolution=res)
        _viewer.track_objects('robot')
        return _viewer