#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse
import sys
import ast

# Other imports.
from simple_rl.agents import QLearningAgent, RMaxAgent, RandomAgent
from simple_rl.tasks import GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldStateClass import GridWorldState
from simple_rl.run_experiments import run_agents_on_mdp
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "This is a Q-Learning agent demo for simple_RL. Pass in the parameters \
        to generate the grid world MDP. Run -h or --help to see what the \
        parameters do. You pass in different modes to visualize the learning of \
        the agent. Passing in mode as `learning` will show the agent learn on \
        the environment. Policy will run VI on the mdp and display the planned \
        policy. agent will run Q learning with a random agent and compare with \
        a random agent.")

    # Add the relevant arguments to the argparser
    parser.add_argument(
        '-w',
        '--width',
        type=int,
        nargs="?",
        const=10,
        default=10,
        help=
        'an integer representing the number of cells for the GridWorld width')
    parser.add_argument(
        '-H',
        '--height',
        type=int,
        nargs="?",
        const=10,
        default=10,
        help=
        "an integer representing the number of cells for the GridWorld height")
    parser.add_argument(
        '-s',
        '--slip',
        type=float,
        nargs="?",
        const=0.05,
        default=0.05,
        help=
        "a float representing the probability that the agent will 'slip' and \
        not take the intended action but take a random action at uniform \
        instead")
    parser.add_argument(
        '-il',
        '--i_loc',
        type=ast.literal_eval,
        nargs="?",
        const=(1, 1),
        default=(1, 1),
        help=
        "a tuple of integers representing the starting cell location of the \
        agent, with one- indexing. For example, do -il '(1,1)' , be sure to \
        include apostrophes (unless you use Windows) or argparse will fail!")
    parser.add_argument(
        '-gl',
        '--g_loc',
        type=ast.literal_eval,
        nargs="?",
        const=[(10, 8)],
        default=[(10, 8)],
        help=
        "a list of tuples of of integer-valued coordinates where the agent will \
        receive a large reward and enter a terminal state. Each coordinate is a \
        location on the grid with one- indexing. For example, do -gl '[(3,3)]', \
        be sure to include apostrophes (unless you use Windows) or argparse \
        will fail!")
    parser.add_argument(
        '-ll',
        '--l_loc',
        type=ast.literal_eval,
        nargs="?",
        const=[(9,1), (8,5), (2,5), (8,3), (1,8)],
        default=[(9,1), (8,5), (2,5), (8,3), (1,8)],
        help=
        "a list of tuples of of integer-valued coordinates where the agent will \
        receive a large negative reward and enter a terminal state. Each \
        coordinate is a location on the grid with one-indexing. For example, do \
        -ll '[(3,2)]' , be sure to include apostrophes (unless you use Windows) \
        or argparse will fail!")
    parser.add_argument(
        '-W',
        '--Walls',
        type=ast.literal_eval,
        nargs="?",
        const=[(2, 1), (3, 1), (4, 1), (10, 1), (6, 2), (7, 2), (8, 2), (9, 2),
               (10, 2), (1, 3), (2, 3), (4, 3), (1, 4), (2, 4), (4, 4), (5, 4),
               (6, 4), (10, 4), (1, 5), (4, 5), (5, 5), (6, 5), (7, 5), (9, 5),
               (10, 5), (1, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
               (10, 6), (1, 7), (7, 7), (8, 7), (9, 7), (10, 7), (4, 8), (5, 8),
               (7, 8), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (7, 9), (9, 9),
               (10, 9), (1, 10), (2, 10), (3, 10), (9, 10), (10, 10)],
        default=[(2, 1), (3, 1), (4, 1), (10, 1), (6, 2), (7, 2), (8, 2), (9, 2),
               (10, 2), (1, 3), (2, 3), (4, 3), (1, 4), (2, 4), (4, 4), (5, 4),
               (6, 4), (10, 4), (1, 5), (4, 5), (5, 5), (6, 5), (7, 5), (9, 5),
               (10, 5), (1, 6), (4, 6), (5, 6), (6, 6), (7, 6), (8, 6), (9, 6),
               (10, 6), (1, 7), (7, 7), (8, 7), (9, 7), (10, 7), (4, 8), (5, 8),
               (7, 8), (1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (7, 9), (9, 9),
               (10, 9), (1, 10), (2, 10), (3, 10), (9, 10), (10, 10)],
        help="a list of tuples of of integer-valued coordinates where there are \
        'walls' that the agent can't transition into. Each coordinate is a \
        location on the grid with one-indexing.  For example, do -W '[(3,2)]', \
        be sure to include apostrophes (unless you use Windows) or argparse \
        will fail!")
    parser.add_argument(
        '-g',
        '--gamma',
        type=float,
        nargs="?",
        const=0.95,
        default=0.95,
        help='a float representing the decay factor for environment')
    parser.add_argument(
        '-ag',
        '--agents',
        type=ast.literal_eval,
        nargs="?",
        const=['q_learning', 'random', 'rmax', 'potential_q'],
        default=['q_learning', 'random', 'rmax', 'potential_q'],
        help=
        'a list of strings representing agents to run on, pick between q_learning \
                potential_q, rmax and random')
    parser.add_argument('-dq',
                        '--default_q',
                        type=float,
                        nargs="?",
                        const=0.0,
                        default=0.0,
                        help='The float representing the default q_init value')
    parser.add_argument(
        '-cq',
        '--custom_q',
        type=ast.literal_eval,
        nargs="?",
        const={},
        default={},
        help=
        "a nested dictionary of the form {'(x,y)': {'action': value}}. Valid states are \
                (x,y) tuples representing the x,y cell of the gridworld (remember one indexed) \
                 and valid actions are 'right', 'left', 'up', 'down'. Values must be a float"
    )

    args = parser.parse_args()
    return args


def generate_MDP(width, height, init_loc, goal_locs, lava_locs, gamma, walls,
                 slip_prob):
    """ Creates an MDP object based on user input """
    actual_args = {
        "width": width,
        "height": height,
        "init_loc": init_loc,
        "goal_locs": goal_locs,
        "lava_locs": lava_locs,
        "gamma": gamma,
        "walls": walls,
        "slip_prob": slip_prob,
        "lava_cost": 1.0,
        "step_cost": 0.1
    }
    return GridWorldMDP(**actual_args)


def parse_custom_q_table(q_dict, default_q):
    custom_q = defaultdict(lambda: defaultdict(lambda: default_q))
    for state, action_dict in q_dict.items():
        for action, value in action_dict.items():
            custom_q[GridWorldState(*ast.literal_eval(state))][action] = value
    return custom_q


def main(open_plot=True):

    # Setup MDP.

    args = parse_args()
    mdp = generate_MDP(args.width, args.height, args.i_loc, args.g_loc,
                       args.l_loc, args.gamma, args.Walls, args.slip)

    # Initialize the custom Q function for a q-learning agent. This should be
    # equivalent to potential shaping.
    # This should cause the Q agent to learn more quickly.

    custom_q = parse_custom_q_table(args.custom_q, args.default_q)

    agents = []
    for agent in args.agents:
        if agent == 'q_learning':
            agents.append(QLearningAgent(actions=mdp.get_actions()))
        elif agent == 'potential_q':
            agents.append(
                QLearningAgent(actions=mdp.get_actions(),
                               custom_q_init=custom_q,
                               name="Potential_Q"))
        elif agent == 'random':
            agents.append(RandomAgent(actions=mdp.get_actions()))
        elif agent == 'rmax':
            agents.append(RMaxAgent(mdp.get_actions()))

    # Run experiment and make plot.
    run_agents_on_mdp(agents,
                      mdp,
                      instances=1,
                      episodes=20,
                      steps=100,
                      open_plot=open_plot,
                      verbose=True)


if __name__ == "__main__":
    main(open_plot=not sys.argv[-1] == "no_plot")
