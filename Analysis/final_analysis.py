"""
 Goals
 1) Fitness curve of each task along generations.
    - Average and Standard deviation of robot's fitness that were optimized in the same task.
    - 4 Curves
        - baseline-walkerv0
        - baseline-BridgeWalker_v0
        - quadrantv0 (walker)
        - quadrantv0 (bridge)
    - Check:
        - Does quadrant sacrifice fitness to acomodate two tasks? Does quadrand achieve reasonable level of fitness?
 2) Hamming distance: Same task X Different task
    - Different task: How similar are the bots that were optimized for different things.
    - Same task: How similar are the bots that were optimized for the same thing.
    - General: Hamming distance between bots independently of the task they were optimized in.
    - Two ideas:
        - Graph1: Line along generations
            - 6 Curves:      
                - baseline-walkerv0 [general] (theres only one task)
                - baseline-BridgeWalker_v0 [general] (theres only one task)
                - quadrantv0 [general]
                - quadrantv0-walker [same task]
                - quadrantv0-bridge [same task]
                - quadrantv0 [different task]  
            - Each point is the average hamming distance of pairs (different task pairs, same task pairs and any pairs)
            - Check: 
                - Graph 1.1: 3 Curves: Comparison between generals -> How diversity evolves
                - Graph 1.2: 2 Curves: Quad Same task + Quad different task 
                    - If same = different, suggest that tasks are similar, robot converged to one form. probably robust one.
                    - If same != different, suggest that tasks requires different robots, no global conversion.
                - Graph 1.3: Bonus: 2 Curves: Comparison of quads that have the same task.
                    - One figure for each task. One Curve for each quadrand.
                    - Compare robots of one quadrand to another quadrand.
                    - Same task but, did evolution happened the same way? How different are the bots that were optimized for the same task but had frontiers.
        - Graph2: BoxPlot of last generation robots.
            - X axis: Each point is a "curve" of Graph1. 
                - baseline-walkerv0 [general], baseline-BridgeWalker_v0 [general], and so on...
            - Y axis: 
                - Hamming distance of last generation in each seed for all X axis points.
            - Check:
                - Consistency of hamming distance along the seeds for the best bots of each case.
                - Same level of analysis of Graph1.
                    - Graph 2.1: Between generals
                    - Graph 2.2: Inside Quads and same task baseline [for walker]
                    - Graph 2.3: Inside Quads and same task baseline [for bridge]
    - Check
        - Are bots that were optimized in the same task similar? 
        - Did the new grid (quadrand) create differences between robots?

 3) CrossEvaluation: Robustness
 #TODO: Estou fazendo o codigo pros robos serem avaliados na task que nao foram ainda. 
    - Evaluate all bots from the last generation of baseline in the tasks they weren't evaluated.
    - Graph1: Boxplot for each grid of the bots, for each task.
        - Graph 1.1: For WALKER task
            - (6 boxes) X axis: baseline-walkerv0, baseline-BridgeWalker_v0, quadrantv0 [geral], quadrantv0[optimized in walker], quadrantv0[optimized in bridge]
            - Y axis: Robots fitness in walker task.
            - Check (in this task):
                - Does quadrantv0 [geral] produce robots whose fitness are comparale to baseline walker? Are they better than baseline bridge?
                    - Same for quadrantv0[optimized in walker]
                - Do quadrantv0[optimized in bridge] perform better than baseline bridge? How is their fitness compared to bots trained in walker?
                - Robustness comes from bridge being able to perform in walker.
        - Graph 1.2: For BRIDGE task
            - (6 boxes) X axis: baseline-walkerv0, baseline-BridgeWalker_v0, quadrantv0 [geral], quadrantv0[optimized in walker], quadrantv0[optimized in bridge]
            - Y axis: Robots fitness in bridge task.
            - Check (in this task):
                - Does quadrantv0 [geral] produce robots whose fitness are comparale to baseline bridge? Are they better than baseline walker?
                    - Same for quadrantv0[optimized in bridge]
                - Do quadrantv0[optimized in walker] perform better than baseline walker? How is their fitness compared to bots trained in bridge?
                - Robustness comes from walkers being able to perform in bridge.
"""