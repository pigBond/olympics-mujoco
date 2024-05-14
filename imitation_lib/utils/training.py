import os
from copy import deepcopy
import numpy as np

from mushroom_rl.utils.dataset import compute_J, parse_dataset


class BestAgentSaver:

    def __init__(self, save_path, n_epochs_save=10, save_replay_memory=False):
        self.best_curr_agent = None
        self.save_path = save_path
        self.n_epochs_save = n_epochs_save
        self.last_save = 0
        self.epoch_counter = 0
        self.best_J_since_last_save = -float('inf')
        self.save_replay_memory = save_replay_memory

    def save(self, agent, J):

        if self.n_epochs_save != -1:
            if J > self.best_J_since_last_save:
                self.best_J_since_last_save = J
                # if the agent has a replay memory that should not be saved, we can save memory by not copying it,
                # i.e., temporarily removing it from the current agent and then giving it back.
                mem = None
                tmp_store_mem = hasattr(agent, '_replay_memory') and not self.save_replay_memory
                if tmp_store_mem:
                    mem = agent._replay_memory
                    agent._replay_memory = None
                self.best_curr_agent = (deepcopy(agent), J, self.epoch_counter)
                if tmp_store_mem:
                    agent._replay_memory = mem

            if self.last_save + self.n_epochs_save <= self.epoch_counter:
                self.save_curr_best_agent()

            self.epoch_counter += 1

    def save_curr_best_agent(self):

        if self.best_curr_agent is not None:
            path = os.path.join(self.save_path, 'agent_epoch_%d_J_%f.msh' % (self.best_curr_agent[2],
                                                                             self.best_curr_agent[1]))
            self.best_curr_agent[0].save(path, full_save=True)
            self.best_curr_agent = None
            self.best_J_since_last_save = -float('inf')
            self.last_save = self.epoch_counter

    def save_agent(self,  agent, J):
        path = os.path.join(self.save_path, 'agent_J_%f.msh' % J)
        agent.save(path, full_save=True)


def prepare_expert_data(data_path):
    dataset = dict()

    # load expert training data
    expert_files = np.load(data_path)
    dataset["states"] = expert_files["states"]
    dataset["actions"] = expert_files["actions"]
    dataset["episode_starts"] = expert_files["episode_starts"]

    # maybe we have next action and next next state
    try:
        dataset["next_actions"] = expert_files["next_actions"]
        dataset["next_next_states"] = expert_files["next_next_states"]
    except KeyError as e:
        print("Did not find next action or next next state.")

    # maybe we have next states and dones in the dataset
    try:
        dataset["next_states"] = expert_files["next_states"]
        dataset["absorbing"] = expert_files["absorbing"]
    except KeyError as e:
        print("Warning Dataset: %s" % e)

    # maybe we have episode returns, if yes done
    try:
        dataset["episode_returns"] = expert_files["episode_returns"]
        return dataset
    except KeyError:
        print("Warning Dataset: No episode returns. Falling back to step-based reward.")

    # this has to work
    try:
        dataset["rewards"] = expert_files["rewards"]
        return dataset
    except KeyError:
        raise KeyError("The dataset has neither an episode nor a step-based reward!")
