from evogym import EvoWorld, EvoViewer, EvoSim
import numpy as np
import json

# This world loads a line with a few bumps.
# The goal of the robot is to go as far as possible

def get_random(length = 60, freq = 0.1,
               bump_height = 3, bump_length = 10):
    return LineWorld(length = length, freq = freq,
                     bump_height = bump_height, bump_length = bump_length)


def get_fromfile(filename):
    return LineWorld(filename = filename)


class LineWorld():
    world = None
    robot = None
    sim = None

    def __init__(self, filename = None,
                 length = 40, freq = 0.1, bump_height = 2, bump_length = 10):
        # Load from a file or
        # Generate random - line with random square blocks
        if filename is None:
            floor = np.zeros((bump_height+1, length))
            floor[-1,:] = 5
            for i in range(length):
                if (i > 5 and np.random.random() < freq):
                    bl = i + np.random.randint(bump_length + 1)
                    bh = np.random.randint(2, bump_height + 2)
                    floor[-bh:, i:bl] = 5
            self.world = EvoWorld()
            self.world.add_from_array(name = "Floor",
                                      structure = floor, x = 0, y = 0)          
        else:
            self.load_json(filename)


    def load_json(self, filename):
        with open(filename, "r") as in_f:
            data = json.loads(in_f.read())
            if (data["class"] != __name__):
                raise Exception("Invalid File!")
            self.world = EvoWorld()
            self.world.add_from_array(name = "Floor",
                                      structure = np.array(data["floor"]), x = 0, y = 0)

        
                
    def save_json(self, filename):
        with open(filename, "w") as out_f:
            data = {"class": __name__,
                    "floor": np.flip(self.world.grid, axis = 0).tolist()
                }
            json.dump(data,
                      out_f,
                      separators = (',', ':'))
        
        
    def set_robot(self, robot):
        self.robot = robot
        self.world.add_from_array(
            name = 'robot',
            structure = robot.shape,
            x = 0, y = 1)

    def reset(self):
        if self.robot is None:
            raise Exception("Can't reset world: No robot set!")

        self.sim = EvoSim(self.world)
        self.sim.reset()

    def restart(self):
        self.load_json(self.world_file)
        self.sim = None

    def clear_robot(self):
        self.robot = None
        self.world.remove_object("robot")
        

    def step(self):
        if self.sim is None:
            raise Exception("Can't step the world before .reset()'in it!")

        action = self.robot.action(self.sim.get_time())
        self.sim.set_action('robot', action)
        self.sim.step()

    def get_score(self):
        robotpos = self.sim.object_pos_at_time(self.sim.get_time(), "robot")
        score = np.mean(robotpos, axis=1)[0]
        return score

    def pprint(self):
        self.world.pretty_print()

    def get_viewer(self, res = (400, 200)):
        _viewer = EvoViewer(self.sim, resolution = res)
        _viewer.track_objects('Floor')
        return _viewer
