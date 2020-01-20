from simple_rl.planning.PartialOrderPlannerClass import PartialOrderPlanner
from simple_rl.tasks.monkey_playroom.MonkeyPlayroomPOMDPClass import MonkeyPlayroomPOMDP
# from simple_rl.tasks.maze_1d.Maze1DPOMDPClass import Maze1DPOMDP

pomdp = MonkeyPlayroomPOMDP(4)
# pomdp = Maze1DPOMDP()
planner = PartialOrderPlanner(pomdp)
planner.plan()

