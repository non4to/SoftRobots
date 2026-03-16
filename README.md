# SoftRobots — Multi-Task Cellular Genetic Algorithm for Soft Robot Evolution

Research on evolving soft robot morphologies using **Quality-Diversity** algorithms
in [Evolution Gym](https://evolutiongym.github.io). Built on top of [code](https://codeberg.org/caranha/YASRE)
originally created by [Claus Aranha](https://conclave.cs.tsukuba.ac.jp) (University of Tsukuba).

---

## Research Contribution

This fork expands the original one to consider two lines of research: Cellular Genetical Algorithms & Quality-Diversity / MAP-Elites

**The key idea:** Instead of a single shared task, each region of the spatial grid is assigned a
different EvoGym task. Robots are evaluated exclusively on their local task. The hypothesis is that
*different tasks in a grid creates a preassure towards diversity*, producing robots with different bodies
in different grid regions — without any explicit diversity objective.

```
Grid (10×10 example):

  col 0-4           col 5-9
┌──────────────┬─────────────────┐
│   Walker     │  BridgeWalker   │  rows 0-4
├──────────────┼─────────────────┤
│   Walker     │  BridgeWalker   │  rows 5-9
└──────────────┴─────────────────┘

Each cell holds one robot, evaluated on its region's task.
Neighbor selection uses rank-based fitness of neighbors evaluated on the chosen cell's task.
```

### Design Decisions

| Decision                                | Rationale                                                                                                |
| -----------------------------------------| ----------------------------------------------------------------------------------------------------------|
| Open-loop sine controller               | Evolution acts exclusively on the body (morphology); controller is fixed.                                |
| Fitness cache per task                  | Controller is deterministic — same morphology + same world = same score, always.                         |
| Spatial regions (not random assignment) | Creates smooth gradients at region boundaries, potentially producing generalist morphologies at borders. |
---

### `Generators/cga.py` — The Algorithm

The update cycle per generation:

1. **For each cell `(x,y)`:** get Moore neighborhood (8 neighbors)
2. **`select()`:** check fitness cache for each neighbor on the local task.
   Evaluate uncached neighbors. Rank-select a parent using geometric weights.
3. **Crossover + mutation** between cell occupant and selected neighbor
4. **Evaluate child** on local task
5. **Replace** cell occupant only if child fitness ≥ parent fitness (elitist)

---

## Usage

### Run an experiment

```bash
python3 main.py
# reads parameters.json — configure your experiment there
```

### Visualize a result

```bash
python3 Visualize.py log/<run_folder>/_world.json log/<run_folder>/robot_x_y_genN.json
python3 Visualize.py log/<run_folder>/_world.json log/<run_folder>/robot_x_y_genN.json -S  # on screen
```

---

## Robot Representation

`SinRobot` is a 5×5 voxel grid with 5 possible voxel types:

| Value | Type | Behavior |
|---|---|---|
| 0 | Empty | void |
| 1 | Rigid | structural, no movement |
| 2 | Soft | deformable, no actuation |
| 3 | Horizontal Actuator | contracts/expands horizontally |
| 4 | Vertical Actuator | contracts/expands vertically |

Validity requires connectivity and at least one actuator (3 or 4).

The controller is a **fixed open-loop sine wave** — actuation is a function of time only,
with no sensor feedback. This means evolution acts purely on **morphology**.

---

## About

**Fork author:** Felipe Nonato Cardoso Sobral Junior
PhD Student, Evolutionary Computation Laboratory, University of Tsukuba
[non4to.github.io](https://non4to.github.io) · [GitHub: non4to](https://github.com/non4to)

**[Original repository](https://codeberg.org/caranha/YASRE):** [Claus Aranha](https://conclave.cs.tsukuba.ac.jp), University of Tsukuba 