import os
import numpy as np

from PIL import Image
import imageio
from pygifsicle import optimize

from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot

max_step = 100

world = EvoWorld.from_json('test/test_env.json')

robot_structure, robot_connections = sample_robot((5, 5))

# print(robot_structure)
# print(robot_connections)

world.add_from_array(
	name='robot',
	structure=robot_structure,
	x=3,
	y=1,
#	connections=robot_connections  # assumed to be fully connected
)

sim = EvoSim(world)
sim.reset()

viewer = EvoViewer(sim, resolution = (400,200))

# RGB image matrix
# img = viewer.render(mode='img')

#viewer.track_objects('robot')
viewer.track_objects('robot', 'Floor')

print(sim.get_dim_action_space('robot'))

steps = 0
input_size = sim.get_dim_action_space('robot')

#print(sim.get_dim_action_space('Ball'))
fork_size = sim.get_dim_action_space('Fork')

imgs = []

def action_unif(size, steps):
  return np.random.uniform(
           low = 0.6,
           high = 1.6,
           size = (size,)
  	 )

def action_sin(size, steps):
  action = []
  for _ in range(size):
    action.append(np.sin(steps/3 + (_*0.1))+1)
  return np.array(action)

for steps in range(max_step):
  action = action_sin(input_size, steps)
  fork_action = action_sin(fork_size, steps)
  # print("{}: {}".format(steps, action))
  sim.set_action('robot', action)
  sim.set_action('Fork', fork_action)
  sim.step()
  img = viewer.render(mode='img')
  imgs.append(img)

imageio.mimsave('test.gif', imgs, duration=30)
optimize('test.gif')
