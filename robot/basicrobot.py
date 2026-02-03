import json
import numpy as np
from evogym import is_connected

def get_random(w = 5, h = 5):
    r = SinRobot()
    r.randomize(w, h)
    return r

def get_fromfile(filename):
    r = SinRobot()
    r.load_json(filename)
    return r

class SinRobot:
    def __init__(self):
        self.shape = np.array([[1]])

    def valid(self):
        return (is_connected(self.shape) and
                (3 in self.shape or 4 in self.shape))

    def save_json(self, filename):
        with open(filename, "w") as out_f:
            data = {"class": __name__, "shape": self.shape.tolist()}
            json.dump(data,
                      out_f,
                      separators = (',', ':'))

    def load_json(self, filename):
        with open(filename, "r") as in_f:
            rdata = json.loads(in_f.read())
            if rdata["class"] != __name__:
                raise Exception("Invalid File!")
            self.shape = np.array(rdata["shape"])

    def copy(self):
        _new = SinRobot()
        _new.shape = self.shape.copy()
        return _new
            
    def randomize(self, w = 5, h = 5):
        count = 0;
        while True:
            self.shape = np.random.randint(0,5,(w,h))
            if self.valid():
                break
            count += 1
            if (count > 5000):
                raise Exception("Can't find a valid random robot after 5000 tries!")

    def count_actuators(self):
        count = 0
        for _x in self.shape.flatten():
            if _x == 3 or _x == 4:
                count += 1

        return count

    def action(self, steps):
        action = []
        for _ in range(self.count_actuators()):
            action.append(np.sin(steps/3 + (_*0.1))+1)
        return np.array(action)

    def mutate(self, size = 1):
        for _ in range(size):
            count = 0
            while True:
                old_shape = self.shape.copy()
                pos = tuple(np.random.randint(0,5,2))
                self.shape[pos] = np.random.randint(0,5)
                if self.valid():
                    break

                self.shape = old_shape
                count += 1
                if count > 5000:
                    raise Exception("Can't find a valid mutation after 5000 tries!")
                
    def crossover(self, mate):
        count = 0

        while True:
            count += 1
            child1 = self.copy()
            child2 = self.copy()

            pos = np.random.randint(0,4)

            for i in range(5):
                if i > pos:
                    for j in range(5):
                        child1.shape[(i,j)] = self.shape[(i,j)]
                        child2.shape[(i,j)] = mate.shape[(i,j)]
                else:
                    for j in range(5):
                        child1.shape[(i,j)] = mate.shape[(i,j)]
                        child2.shape[(i,j)] = self.shape[(i,j)]

            if child1.valid():
                return child1
            if child2.valid():
                return child2

            if count > 5000:
                return self.copy()
