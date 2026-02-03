import os, sys
import importlib
import json
import numpy as np
from optparse import OptionParser

import imageio
from pygifsicle import optimize

def main():
  options, args = parse_args()

  world_f = args[0]
  robot_f = args[1]

  with open(world_f, "r") as in_f:
    rdata = json.loads(in_f.read())
    world_m = importlib.import_module(rdata["class"])

  with open(robot_f, "r") as in_f:
    rdata = json.loads(in_f.read())
    robot_m = importlib.import_module(rdata["class"])
    
  robot = robot_m.get_fromfile(robot_f)
  world = world_m.get_fromfile(world_f)

  world.set_robot(robot)
  world.reset()

  on_screen = options.screen
  if options.resolution is None:
    viewer = world.get_viewer()   # res = (w,h)
  else:
    viewer = world.get_viewer(res = options.resolution)
  frames = []

  for _step in range(options.sim_step):
    world.step()

    if on_screen:
      viewer.render(mode='screen')
    else:
      frames.append(viewer.render(mode="img"))

  if not on_screen:
    imageio.mimsave(options.filename, frames, duration=20)
    optimize(options.filename)

  print(f"Score: {world.get_score()}")

def parse_args():
  usage = "usage: %prog [options] <world file> <robot file>"
  desc = """Render a specific robot working on a specific world. Display the
render as a gif or on the display.
"""
  import world, robot
  
  parser = OptionParser(usage = usage, description = desc) 

  parser.add_option("-s", "--sim_step",
                    type="int", default = 400,
                    help="Number of Simulation Steps")

  parser.add_option("-o", "--output", default = "animation.gif",
                    type="string", dest="filename",
                    help="Output File for Gif")

  parser.add_option("-S", "--screen",
                    action="store_true", default = False,
                    help="Shows the animation on the screen, instead of a gif")

  parser.add_option("-r", "--resolution",
                    type="int", nargs=2,
                    help="Image Resolution")
    
  # Parser option: 
  
  options, args = parser.parse_args()

  if len(args) != 2:
    parser.error("Requires 2 arguments: World file, Robot file")

  # TODO -- check if files are valid


  return options, args


if __name__ == "__main__":
  main()
