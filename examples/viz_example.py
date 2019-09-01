#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import argparse

# Other imports.
import srl_example_setup
from simple_rl.agents import QLearningAgent, RMaxAgent
from simple_rl.run_experiments import run_single_agent_on_mdp 
from simple_rl.tasks import FourRoomMDP, TaxiOOMDP, GridWorldMDP
from simple_rl.tasks.grid_world.GridWorldMDPClass import make_grid_world_from_file
from simple_rl.planning import ValueIteration

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

def parse_args():
    # Add all arguments
    parser = argparse.ArgumentParser()

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
    parser.add_argument("-vz", '--visualization', type=str, default="learning", nargs='?', help="Choose the visualization type (one of {value, policy, agent, learning or interactive}).")
    parser.add_argument('-a', '--alpha', type=float, default=0.1, nargs='?', help="Learning rate")
    parser.add_argument('-e', '--epsilon', type=float, default=0.1, nargs='?', help='Choosing how much to explore vs exploit')
    parser.add_argument('-an', '--anneal', type=bool, default=False, nargs='?', help="CHoose to anneal or not")
    parser.add_argument('-ex', '--explore', type=str, default="uniform", nargs='?', help="Choose one uniform or softmax for exploration")
    return args

def main():
    
    # Setup MDP, Agents.
    args = parse_args()
    mdp = generate_MDP(
        args.width, 
        args.height,
        args.i_loc,
        args.g_loc,
        args.l_loc,
        args.gamma, 
        args.Walls, 
        args.slip)
    mdp = GridWorldMDP(width=7, height=7, init_loc=(1, 1), goal_locs=[(7, 7)], lava_locs=[(4, 2)], gamma=0.95, walls=[(2, 2)], slip_prob=0.1)
    ql_agent = QLearningAgent(mdp.get_actions(), epsilon=args.epsilon, alpha=args.alpha, gamma=args.gamma, explore=args.explore, anneal=args.anneal) 

    # Choose viz type.

    viz = args.visualization

    if viz == "value":
        # --> Color corresponds to higher value.
        # Run experiment and make plot.
        mdp.visualize_value()
    elif viz == "policy":
        # Viz policy
        value_iter = ValueIteration(mdp)
        value_iter.run_vi()
        policy = value_iter.policy
        mdp.visualize_policy(policy)
    elif viz == "agent":
        # --> Press <spacebar> to advance the agent.
        # First let the agent solve the problem and then visualize the agent's resulting policy.
        print("\n", str(ql_agent), "interacting with", str(mdp))
        run_single_agent_on_mdp(ql_agent, mdp, episodes=500, steps=200)
        mdp.visualize_agent(ql_agent)
    elif viz == "learning":
        mdp.visualize_learning(ql_agent, delay=0.005, num_ep=500, num_steps=200)
    elif viz == "interactive":
        # Press <1>, <2>, <3>, and so on to execute action 1, action 2, etc.
    	mdp.visualize_interaction()

if __name__ == "__main__":
    main()
