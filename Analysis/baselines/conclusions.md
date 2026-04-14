# Things noted from the figures in this folder

- These graphs shows how many bots were produced (in total) by each baseline task (baseline-walkerv0 and 'baseline-BridgeWalker_v0')
- Each graph focus on one characteristic.
    - Number of horizontal actuators;
    - Number of vertical actuators;
    - Horizontal simmetry score;
    - Vertical simmetry score;
    - Height
    - Width

## Width and Height
    - No relevant variation. Most (if not all) robots have width = 5
    - This happens because of how robots are created:
        - Same chance to chose any blocks, only one option is blockless
        - Very easy to come out with invalid robots, which are discarded.

## Vertical and Horizontal Simmetry
    - No relevant difference.
    - Robots with less simmetry seem to be valued by both tasks the better the fit gets.

## Quantity of horizontal actuators
    - When fitness > 0.75
        - More robots with 6 h_act in Walker
        - More robots with 10 h_act in Bridge

    - When fitness > 0.85
        - More robots with 11 h_act in both
    
    - When fitness > 0.9
        - All robots have 11 h_act

    - So robots tend to converge to a similar number of horizontal actuator number.

## Quantity of vertical actuators
    - When fitness > 0.75
        - More robots with less actuators (1,2,3,4) in Walker
            - Minimum = 1, Maximum = 8
        - More robots with more actuators (5) in Bridge
            - Minimum = 3, Maximum = 12

    - When fitness > 0.85
        - Walker has bots with 2, 3 and 6 actuators
        - Bridge has bots with 4, 5, 6 and 8 actuators (most 5)
    
    - When fitness > 0.9
        - Walker has bots with 3 actuators
        - Bridge has bots with 4 actuators

    - So althought there is a difference, bridge gets bots with more vertical actuators, the difference in the top isn't a lot. The best robots have a similar number of actuators.
