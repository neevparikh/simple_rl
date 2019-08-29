#!/usr/bin/env python

# Python imports
import matplotlib as plt
import numpy as np
import time
import argparse
import ast

# Other imports
import srl_example_setup
from simple_rl.tasks import GridWorldMDP
from simple_rl.planning.ValueIterationClass import ValueIteration

# INSTRUCTIONS FOR USE:
# 1. When you run the program [either with default or supplied arguments], a pygame window should pop up.
#    This is the first iteration of running VI on the given MDP.
# 2. Press escape to close this pygame window and wait, another window will pop-up displaying the 
#    policy from the next time step. Press q to exit. 
# 3. Repeat 1 and 2 until the program terminates

# An input function, creates the MDP object based on user input
def generate_MDP(width, height, init_loc, goal_locs, lava_locs, gamma, walls, slip_prob):
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
    # This accepts arguments from the command line with flags.
    # Example usage: python value_iteration_update_viz_example.py -w 7 -H 5 -s 0.05 -g 0.95 -il '(1,1)' -gl '[(7,4)]' -ll '[(7,3)]' -W '[(2,2)]'
    parser = argparse.ArgumentParser(description='Run a demo that shows a visualization of value' +  
                                                 'iteration on a GridWorld MDP. \n Notes: \n 1.' + 
                                                 'Goal states should appear as green circles, lava' +  
                                                 ' states should be red circles and the agent start' + 
                                                 ' location should appear with a blue triangle. If' + 
                                                 ' these are not shown, you have probably passed in' + 
                                                 ' a value that is outside the grid \n 2.' + 
                                                 'This program is intended to provide a visualization' +
                                                 ' of Value Iteration after every iteration of the algorithm.' + 
                                                 ' Once you pass in the correct arguments, a PyGame screen should pop-up.' +
                                                 ' Press the esc key to view the next iteration and the q key to quit' + 
                                                 '\n 3. The program prints the total time taken for VI to run in seconds ' + 
                                                 ' and the number of iterations (as the history) to the console.')

    # Add the relevant arguments to the argparser
    parser.add_argument('-w', '--width', type=int, nargs="?", const=4, default=4,
    help='an integer representing the number of cells for the GridWorld width')
    parser.add_argument('-H', '--height', type=int, nargs="?", const=3, default=3,
    help='an integer representing the number of cells for the GridWorld height')
    parser.add_argument('-s', '--slip', type=float, nargs="?", const=0.05, default=0.05,
    help='a float representing the probability that the agent will "slip" and not take the intended action but take a random action at uniform instead')
    parser.add_argument('-g', '--gamma', type=float, nargs="?", const=0.95, default=0.95,
    help='a float representing the decay factor for Value Iteration')
    parser.add_argument('-il', '--i_loc', type=ast.literal_eval, nargs="?", const=(1,1), default=(1,1),
    help="a tuple of integers representing the starting cell location of the agent, with one-indexing. For example, do -il '(1,1)' , be sure to inclue apostrophes or argparse will fail!")
    parser.add_argument('-gl', '--g_loc', type=ast.literal_eval, nargs="?", const=[(3,3)], default=[(3,3)],
    help="a list of tuples of of integer-valued coordinates where the agent will receive a large reward and enter a terminal state. Each coordinate is a location on the grid with one-indexing. For example, do -gl '[(3,3)]' , be sure to inclue apostrophes or argparse will fail!")
    parser.add_argument('-ll', '--l_loc', type=ast.literal_eval, nargs="?", const=[(3,2)], default=[(3,2)],
    help="a list of tuples of of integer-valued coordinates where the agent will receive a large negative reward and enter a terminal state. Each coordinate is a location on the grid with one-indexing. For example, do -ll '[(3,2)]' , be sure to inclue apostrophes or argparse will fail!")
    parser.add_argument('-W', '--Walls', type=ast.literal_eval, nargs="?", const=[(2,2)], default=[(2,2)],
    help="a list of tuples of of integer-valued coordinates where there are 'walls' that the agent can't transition into. Each coordinate is a location on the grid with one-indexing. For example, do -W '[(3,2)]' , be sure to inclue apostrophes or argparse will fail!")
    parser.add_argument('-d', '--delta', type=float, nargs="?", const=0.0001, default=0.0001,
    help='After an iteration if VI, if no change more than delta has occurred, terminates.')
    parser.add_argument('-m', '--max-iter', type=int, nargs="?", const=500, default=500,
    help='Maximum number of iterations VI runs for')

    args = parser.parse_args()
    mdp = generate_MDP(
        args.width, 
        args.height,
        args.i_loc,
        args.g_loc,
        args.l_loc,
        args.gamma, 
        args.Walls, 
        args.slip)

    # Run value iteration on the mdp and save the history of value backups until convergence
    st = time.time()
    vi = ValueIteration(mdp, max_iterations=args.max_iter, delta=args.delta)
    _, _, histories = vi.run_vi_histories()
    end = time.time()

    print('Took {:.4f} seconds'.format(end-st))

    # For every value backup, visualize the policy
    num_hist = len(histories)
    for i, value_dict in enumerate(histories):
        print('Showing history {:04d} of {:04d}'.format(i+1, num_hist))
        mdp.visualize_policy(lambda in_state: value_dict[in_state]) # Note: This lambda is necessary because the policy must be a function

if __name__ == "__main__":
    main()
