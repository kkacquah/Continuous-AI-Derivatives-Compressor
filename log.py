import os
import datetime
import numpy as np
import shutil

class ExperimentLog():
    def __init__ (self):
        self.excess_percents = []
        self.rewards = []
        self.step_excess_percents = []
        self.step_rewards = []
        self.time = str(datetime.datetime.now())
        self.path = 'C:\\tmp\\gym_compressor_logs'
        #delete previous experiment log
        if os.path.exists(self.path):
            shutil.rmtree(self.path)
        os.makedirs(self.path)
    def observe_episode(self, excess_percent, reward):
        self.excess_percents += [excess_percent]
        self.rewards += [reward]
    def reset_episode(self):
        self.step_excess_percents = self.step_excess_percents + [[]]
        self.step_rewards =  self.step_rewards + [[]]
    def observe_step(self,excess_percent, reward):
        self.step_excess_percents[-1] = self.step_excess_percents[-1] + [excess_percent]
        self.step_rewards[-1] = self.step_excess_percents[-1] + [reward]
def save(name, log: ExperimentLog):
    out_prefix = log.path
    if not os.path.exists(out_prefix):
        os.makedirs(out_prefix)
    filename_episode = name + "_episode_level_" + log.time.replace(' ','_').replace(':','.') + ".csv"
    full_out_path = os.path.join(out_prefix, filename_episode)
    data = np.c_[(log.excess_percents, log.rewards)]
    np.savetxt(full_out_path, data,
               delimiter=",")
    filename_step_ep = name + "_step_level_ep_" + log.time.replace(' ','_').replace(':','.') + ".csv"
    full_out_path = os.path.join(out_prefix, filename_step_ep)
    data = np.array(log.step_excess_percents).T
    np.savetxt(full_out_path, data,
               delimiter=",")
    filename_step_r = name + "_step_level_r_" + log.time.replace(' ','_').replace(':','.') + ".csv"
    full_out_path = os.path.join(out_prefix, filename_step_r)
    data = np.array(log.step_rewards).T
    np.savetxt(full_out_path, data,
               delimiter=",")