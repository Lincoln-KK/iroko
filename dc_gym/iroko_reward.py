from __future__ import print_function
import numpy as np


class RewardFunction:
    def __init__(self, topo_conf, reward_model, stats_dict):
        self.sw_ports = topo_conf.get_sw_ports()
        self.host_ports = topo_conf.get_host_ports()
        self.max_queue = topo_conf.max_queue
        self.max_bps = topo_conf.max_bps
        self.num_sw_ports = topo_conf.get_num_sw_ports()
        self.reward_model = reward_model
        self.stats_dict = stats_dict

    def get_reward(self, stats, deltas, actions):
        reward = 0
        if "action" in self.reward_model:
            action_reward = self._action_reward(actions)
            # print("action: %f " % action_reward, end='')
            reward += action_reward
        if "bw" in self.reward_model:
            bw_reward = self._bw_reward(stats)
            # print("bw: %f " % bw_reward, end='')
            bw_reward = self._adjust_reward(bw_reward, deltas)
            reward += bw_reward
        if "backlog" in self.reward_model:
            queue_reward = self._queue_reward(stats)
            reward += queue_reward
            # print("queue: %f " % queue_reward, end='')
        if "joint_backlog" in self.reward_model:
            joint_queue_reward = self._joint_queue_reward(actions, stats)
            reward += joint_queue_reward
            # print("joint queue: %f " % joint_queue_reward, end='')
        if "std_dev" in self.reward_model:
            std_dev_reward = self._std_dev_reward(actions)
            reward += std_dev_reward
            # print("std_dev: %f " % std_dev_reward, end='')
        if "fairness" in self.reward_model:
            fairness_reward = self.fairness(actions)
            reward += fairness_reward
            # print("fairness: %f " % fairness_reward, end='')
        # print("Total: %f" % reward)
        return reward

    def _adjust_reward(self, reward, queue_deltas):
        if "olimit" in self.reward_model:
            tmp_list = []
            for port_stats in queue_deltas:
                tmp_list.append(port_stats[self.stats_dict["olimit"]])
            if any(tmp_list):
                reward /= 4
        if "drops" in self.reward_model:
            tmp_list = []
            for port_stats in queue_deltas:
                tmp_list.append(port_stats[self.stats_dict["drops"]])
            if any(tmp_list):
                reward /= 4
        return reward

    def _std_dev_reward(self, actions):
        return -np.std(actions)

    def _fairness_reward(self, actions):
        '''Compute Jain's fairness index for a list of values.
        See http://en.wikipedia.org/wiki/Fairness_measure for fairness equations.
        @param values: list of values
        @return fairness: JFI
        '''
        num = sum(actions) ** 2
        denom = len(actions) * sum([i ** 2 for i in actions])
        return num / float(denom)

    def _action_reward(self, actions):
        return np.average(actions)

    def _bw_reward(self, stats):
        bw_reward = []
        for index, iface in enumerate(self.sw_ports):
            if iface in self.host_ports:
                bw_reward.append(stats[self.stats_dict["bw_rx"]][index])
        return np.average(bw_reward)

    def _queue_reward(self, stats):
        queue_reward = 0.0
        weight = float(self.num_sw_ports) / float(len(self.host_ports))
        for index, _ in enumerate(self.sw_ports):
            queue_reward -= stats[self.stats_dict["backlog"]][index]**2
        return queue_reward * weight

    def _joint_queue_reward(self, actions, stats):
        queue_reward = 0.0
        # weight = float(self.num_sw_ports) / float(len(self.host_ports))
        flip_action_reward = False
        for index, _ in enumerate(self.sw_ports):
            queue = stats[self.stats_dict["backlog"]][index]
            queue_reward -= queue
            if queue > 0.20:
                flip_action_reward = True
        if flip_action_reward:
            queue_reward += (1 - self._action_reward(actions))
        else:
            queue_reward += self._action_reward(actions)
            # queue_reward += self._fairness_reward(actions)
        return queue_reward
