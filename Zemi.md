CELLULAR GENETICAL ALGORITHM, which we are going to call CGA in this presentation, is a type of GA where individuals only interact with individuals within what we'll call a neighborhood. During parent selection with limit the parents available to be choosen to individuals that have a certain proximity. We determine this proximity, this neighborhood by using a grid where each cell has an individual.

Each cell of this grid has a problem to be solved. 
Only one individual occupies each cell and is evaluated on the problem that cell has.
For this first case, we consider all cells to have the same problem.
So let's talk in terms of softrobots to explain the algorithm.

The problem of each cell is a task do be solved by a softrobot.
An individual is a matrix [5x5] that represents the body of a softrobot.
    - Each position of the matrix may assume a number between 0 and 5, where each number represents a type of block. ('EMPTY': 0, 'RIGID': 1, 'SOFT': 2, 'H_ACT': 3,'V_ACT': 4, 'FIXED': 5)
    - Robots must always be connected to be valid.
    - Robots actuators have a deterministic controler: constant, they all have the same. Differences in behavior are determined only by body shape.
    - This shape matrix is what evolves, it is this that goes through crossover and mutation.
    - An individual fitness is how far right he went. This value can be negative! But we normalize it so we can see it better. 1 = max distance run, 0 = min distance run (generally a bot that run left lol.)
A neighborhood is represented by its 8 neighrbors in the 8 cardinal directions (moore neighborhood)

Now the algorithm.
We first create a grid that is 10x10 and assign a task to each cell.
We then generate one robot for each cell and evaluate them in the task of that cell. <- INICIALIZATION
Now we get into the algorithm loop. For a number of generations we repeat this:
    - For each cell, we choose the robot in that cell as parent1 and do a parent selection for parent2
        - [PARENT2 SELECTION]:
            - Evaluate all neighbors in parent1 task.
            - Rank neighbors according to their fitness in parent1 task
            - Higher fitness means higher chances of getting chosen
            - Pick parent2
        - Parent1: Bot in focused cell; Parent2: Selected from neighborhood
    - Crossover parent1 with parent2, then mutate child (considering mutation chance)
        - [CROSSOVER]: 
            - Pick a random number between 0 and 4 and cuts matrix in that line. 
            - Part from parent 1, part from parent 2.
        - [MUTATION]:
            - Picks a random position of the matrix and changes the block to a random one
            - Until a valid mutation or X tries
    - Child is evaluated in PARENT1 task. 
    - If child fitness is better, it takes parent1 place in the grid for the next generation.
    - A generation ends when all cells go through this process.

Now. What if cells don't have necessarily the same task? This is what I have been testing now.
Instead of only having a map where we have cell with only task walker, or task bridge.
lets divide the map into quadrants where we alternate the task between quadrants, like this!
We then apply the same algorithm. Whats the difference? None. But look, in the border we will have interactions of robots that were optimized for different tasks.

Look at this cell in the border. When we do parent selection to select parent2, we have some neighbors that were optimized in a different task. These neighbors are evaluated on parent1's task and then assigned a probability to be chosen according to their rank when compared to the other neighrbors. 

From here we have a few outcomes.
1. This guy could be so bad at parent1's task that he may never be chosen to crossover and no gene is shared between different tasks.
2. This guy could very good (or decent) and get chosen. Now we mixed genes from different tasks. From here the hypothesis is that this mix will allow:
    - Robots that are good on both tasks;
        - Since we are mixing robots that were optimized in different tasks.
    - Higher diversity of robot-shapes;
        - Different tasks evolve different robots, their interaction allows them to touch new spaces of the solution space;

Right now I'm working on getting some graphs done to test these things. But i can show some interesting things that i already got. Some of these figures are in that forum at discord, where I intent to make it like a blog and post stuff as a do it.





