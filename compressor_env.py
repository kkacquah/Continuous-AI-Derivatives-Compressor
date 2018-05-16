 #!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplified contract origination environment.
"""

# core modules
from . import log
from . import graph_computations as gc
import numpy as np
from operator import itemgetter
import random
import matplotlib.pyplot as plt
# 3rd party modules
import gym
from gym import spaces

class CompressorEnv(gym.Env):
    """
    Define a simple Compressor environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """
    picks_max = False
    excess_percent_observation = False
    clear_all_cycles = False
    logname = "random"
    def __init__(self):
        self.__version__ = "0.1.0"
        print("CompressorEnv - Version {}".format(self.__version__))
        self.log = log.ExperimentLog()
        n_most_connected = 10
        #Load Derivatives Data into the CompressorEnv
        week_adj_matrices_all,n = gc.load_data_to_adj_matrices('C:\Python36\Lib\site-packages\gym\envs\gym_compressor\complete_credit_data.csv','2000-02-08')
        n_most_connected_list = gc.find_most_connected(week_adj_matrices_all,n,n_most_connected)
        self.week_adj_matrices = gc.keep_n_most_connected(week_adj_matrices_all, n_most_connected_list)
        #Create dictionary with necessary data
        # General variables defining the environment
        self.episode_over = False
        self.curr_step = -1
        self.curr_date = list(self.week_adj_matrices.keys())[self.curr_step]
        self.counterfactual_adj_matrix = self.week_adj_matrices[self.curr_date]
        self.curr_adj_matrix = self.week_adj_matrices[self.curr_date]
        #Right now "is_compressed" is set to true when the current notional is 75% of the counterfactual notional
        self.is_compressed = False
        # Define what the agent can do
        self.action_space = spaces.Discrete(500)
        # Observation consists of the current adjacency matrix, the compressed matrix, excess percent of the matrix
        if self.excess_percent_observation:
            self.observation_space = spaces.Box(low=0, high=1.0, shape=(1,),dtype = np.float32)
        else:
            self.observation_space = spaces.Dict({"curr_adj_matrix":spaces.Box(low=0, high=1.0, shape=(n,n),dtype = np.float32),
                                            "compressed_matrix":spaces.Box(low=0, high=1.0, shape=(n,n),dtype = np.float32),
                                            "excess_percent":spaces.Box(low=0, high=1.0, shape=(1,),dtype = np.float32)})
        # Store what the agent tried
        self.curr_adj_matrix = np.zeros((n,n))
        self.steps = len(list(self.week_adj_matrices.keys()))-1
        self.curr_episode = -1
        self.curr_reward = 0
        self.excess_percent = 1
        self.action_episode_memory = []

    def _step(self, action):
        """
        The agent takes a step in the environment.

        Parameters
        ----------
        action : int

        Returns
        -------
        ob, reward, episode_over, info : tuple
            ob (object) :
                an environment-specific object representing your observation of
                the environment.
            reward (float) :
                amount of reward achieved by the previous action. The scale
                varies between environments, but the goal is always to increase
                your total reward.
            episode_over (bool) :
                whether it's time to reset the environment again. Most (but not
                all) tasks are divided up into well-defined episodes, and done
                being True indicates the episode has terminated. (For example,
                perhaps the pole tipped too far, or you lost your last life.)
            info (dict) :
                 diagnostic information useful for debugging. It can sometimes
                 be useful for learning (for example, it might contain the raw
                 probabilities behind the environment's last state change).
                 However, official evaluations of your agent are not allowed to
                 use this for learning.
        """
        if self.curr_step > self.steps:
            raise RuntimeError("Episode is done")

        

        ob = self._get_state()
        self.curr_step += 1
        self.curr_date = list(self.week_adj_matrices.keys())[self.curr_step]
        self._take_action(action)
        
        if ob["excess_percent"] < 0.60:
            self.is_compressed = True
        else:
            self.is_compressed = False
        reward = self._get_reward()

        if self.curr_step == self.steps:
            self.episode_over = True
        self.log.observe_step(self.excess_percent,reward)
        return ob, reward, self.episode_over, {"excess_percent":self.excess_percent}

    def _take_action(self, action):
        if self.picks_max:
            action = 500
        if self.clear_all_cycles:
            action = len(self.ccycles) - 1
        self.action_episode_memory[self.curr_episode].append(action)
        if action > len(self.ccycles):
            action = len(self.ccycles)
        cycles = self.ccycles[:action]
        if len(cycles) > 0:
            for cycle in cycles:
                self.curr_adj_matrix = gc.compress_critical_cycle(cycle,self.curr_adj_matrix)

    def _get_reward(self):
        """Reward is given for a sold banana."""
        if self.is_compressed:
            return 1
        else:
            return -1

    def _reset(self):
        """
        Reset the state of the environment and returns an initial observation.

        Returns
        -------
        observation (object): the initial observation of the space.
        """

        self.log.observe_episode(self.excess_percent,self.curr_reward)
        log.save(self.logname, self.log)
        self.log.reset_episode()
        self.curr_episode += 1
        self.action_episode_memory.append([])
        self.is_compressed = False
        self.episode_over = False
        self.curr_step = -1
        self.curr_date = list(self.week_adj_matrices.keys())[self.curr_step]
        self.counterfactual_adj_matrix = self.week_adj_matrices[self.curr_date]
        self.curr_adj_matrix = self.week_adj_matrices[self.curr_date]

        
        return self._get_state()

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        self.counterfactual_adj_matrix = self.counterfactual_adj_matrix + self.week_adj_matrices[self.curr_date]
        self.curr_adj_matrix = self.curr_adj_matrix + self.week_adj_matrices[self.curr_date]
        print(np.sum(self.counterfactual_adj_matrix))
        self.excess_percent = np.sum(self.curr_adj_matrix)/np.sum(self.counterfactual_adj_matrix)
        compress_res = gc.compress_data(self.curr_adj_matrix)
        critical_matrix = compress_res["critical_matrix"]
        compressed_matrix = self.curr_adj_matrix - critical_matrix
        self.ccycles = gc.critical_list_from_matrix(critical_matrix)
        if self.excess_percent_observation:
            return self.excess_percent
        else:
            return dict({"curr_adj_matrix":gc.normalize_matrix(self.curr_adj_matrix),
                    "compressed_matrix":gc.normalize_matrix(compressed_matrix),
                    "excess_percent":self.excess_percent})


    def _seed(self, seed):
        random.seed(seed)
        np.random.seed
class CompressorEnvPicksMax(CompressorEnv):
    picks_max = True
    logname = "max"
class CompressorEnvExcessPercent(CompressorEnv):
    excess_percent_observation = True
    logname = "maximum"
class CompressorEnvClearAllCycles(CompressorEnv):
    clear_all_cycles = True
    logname = "all_cycles"
        


