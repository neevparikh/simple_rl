#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse
import ast

# Other imports.
from simple_rl.agents import QLearningAgent, RandomAgent
from simple_rl.run_experiments import run_agents_on_mdp
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning import ValueIteration


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
        '-a',
        '--alpha',
        type=float,
        nargs="?",
        const=0.2,
        default=0.2,
        help='a float representing the learning rate for Q agent')
    parser.add_argument('-ep',
                        '--epsilon',
                        type=float,
                        nargs="?",
                        const=0.1,
                        default=0.1,
                        help='a float representing the epsilon term')
    parser.add_argument('-an',
                        '--anneal',
                        action='store_true',
                        help='to anneal or not in q learning')
    parser.add_argument('-ex',
                        '--explore',
                        type=str,
                        nargs="?",
                        const="uniform",
                        default="uniform",
                        help='a string representing the exploration term')
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        nargs="?",
        const='learning',
        default='learning',
        help='Mode of visualization: one of [value, policy, learning, agent]')

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


def main():

    args = parse_args()
    mdp = generate_MDP(args.width, args.height, args.i_loc, args.g_loc,
                       args.l_loc, args.gamma, args.Walls, args.slip)

    ql_agent = QLearningAgent(mdp.get_actions(),
                              epsilon=args.epsilon,
                              alpha=args.alpha,
                              gamma=args.gamma,
                              explore=args.explore,
                              anneal=args.anneal)
    viz = args.mode

    if viz == "value":
        # --> Color corresponds to higher value.
        # Run experiment and make plot.
        mdp.visualize_value()
    elif viz == "policy":
        # Viz policy
        value_iter = ValueIteration(mdp)
        value_iter.run_vi()
        mdp.visualize_policy_values(
            (lambda state: value_iter.policy(state)),
            (lambda state: value_iter.value_func[state]))
    elif viz == "agent":
        # --> Press <spacebar> to advance the agent.
        # First let the agent solve the problem and then visualize the  
        # agent's resulting policy.
        print("\n", str(ql_agent), "interacting with", str(mdp))
        rand_agent = RandomAgent(actions=mdp.get_actions())
        run_agents_on_mdp([rand_agent, ql_agent],
                          mdp,
                          open_plot=True,
                          episodes=60,
                          steps=200,
                          instances=5,
                          success_reward=1)
        # mdp.visualize_agent(ql_agent)
    elif viz == "learning":
        # --> Press <r> to reset.
        # Show agent's interaction with the environment.
        mdp.visualize_learning(ql_agent,
                               delay=0.005,
                               num_ep=500,
                               num_steps=200)


if __name__ == "__main__":
    main()
