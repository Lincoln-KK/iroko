import os
import sys
import atexit
import ctypes
import logging
from multiprocessing import RawArray, RawValue
import numpy as np
from gymnasium import Env as openAIGym, spaces
# from gym import Env as openAIGym, spaces

#import dc_gym.utils as dc_utils
import utils as dc_utils
#from dc_gym.control.iroko_bw_control import BandwidthController
from control.iroko_bw_control import BandwidthController
#from dc_gym.iroko_sampler import StatsSampler
from iroko_sampler import StatsSampler
#from dc_gym.iroko_traffic import TrafficGen
from iroko_traffic import TrafficGen
#from dc_gym.iroko_state import StateManager
from iroko_state import StateManager
#from dc_gym.utils import TopoFactory
from utils import TopoFactory
#from dc_gym.topos.network_manager import NetworkManager
from topos.network_manager import NetworkManager

from logger import get_logger
log = get_logger(__name__)


FILE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CONF = {
    # Input folder of the traffic matrix.
    "input_dir": f"{FILE_DIR}/inputs/",
    # Which traffic matrix to run. Defaults to the first item in the list.
    "tf_index": 0,
    # Output folder for the measurements during trial runs.
    "output_dir": "results/",
    # When to take state samples. Defaults to taking a sample at every step.
    "sample_delta": 1,
    # Use the simplest topology for tests.
    "topo": "dumbbell",
    # Which agent to use for traffic management. By default this is TCP.
    "agent": "tcp",
    # Which transport protocol to use. Defaults to the common TCP.
    "transport": "tcp",
    # If we have multiple environments, we need to assign unique ids
    "parallel_envs": False,
    # Topology specific configuration (traffic pattern, number of hosts)
    "topo_conf": {},
    # The network features supported by this environment
    "stats_dict": {"backlog": 0, "olimit": 1,
                   "drops": 2, "bw_rx": 3, "bw_tx": 4},
    # Specifies which variables represent the state of the environment:
    # Eligible variables are drawn from stats_dict
    # To measure the deltas between steps, prepend "d_" in front of a state.
    # For example: "d_backlog"
    "state_model": ["backlog"],
    # Add the flow matrix to state?
    "collect_flows": False,
    # Specifies which variables represent the state of the environment:
    # Eligible variables:
    # "action", "queue","std_dev", "joint_queue", "fair_queue"
    "reward_model": ["joint_queue"], # Default was ["step"]. Also ["fair_queue"]
}


def squash_action(action, action_min, action_max):
    action_diff = (action_max - action_min)
    return (np.tanh(action) + 1.0) / 2.0 * action_diff + action_min


def clip_action(action, action_min, action_max):
    """ Truncates the entries in action to the range defined between
    action_min and action_max. """
    return np.clip(action, action_min, action_max)


def sigmoid(action, derivative=False):
    sigm = 1. / (1. + np.exp(-action))
    if derivative:
        return sigm * (1. - sigm)
    return sigm


class DCEnv(openAIGym):
    __slots__ = ["conf", "topo", "traffic_gen", "state_man", "steps",
                 "terminated", "net_man", "input_file", "short_id",
                 "bw_ctrl", "sampler", "reward", "active_rate"]

    def __init__(self, conf={}):
        self.conf = DEFAULT_CONF
        self.conf.update(conf)

        # Init one-to-one mapped variables
        self.net_man = None
        self.state_man = None
        self.traffic_gen = None
        self.bw_ctrl = None
        self.sampler = None
        self.input_file = None
        self.terminated = False
        self.reward = RawValue('d', 0)

        # set the id of this environment
        self.short_id = dc_utils.generate_id()
        if self.conf["parallel_envs"]:
            self.conf["topo_conf"]["id"] = self.short_id
        # initialize the topology
        self.topo = TopoFactory.create(self.conf["topo"],
                                       self.conf["topo_conf"])
        # Save the configuration we have, id does not matter here
        dc_utils.dump_json(path=self.conf["output_dir"],
                           name="env_config", data=self.conf)
        dc_utils.dump_json(path=self.conf["output_dir"],
                           name="topo_config", data=self.topo.conf)
        # set the dimensions of the state matrix
        self._set_gym_matrices()
        # Set the active traffic matrix
        self._set_traffic_matrix(
            self.conf["tf_index"], self.conf["input_dir"], self.topo)
        # Default sets self.input_file to "incast_2" for dumbbell & "stag_prob_0_2_3_data" for fat-tree 
        
        # each unique id has its own sub folder
        if self.conf["parallel_envs"]:
            self.conf["output_dir"] += f"/{self.short_id}"
        # check if the directory we are going to work with exists
        dc_utils.check_dir(self.conf["output_dir"])

        # handle unexpected exits scenarios gracefully
        atexit.register(self.close)

    def _set_gym_matrices(self):

        # set the action space
        num_actions = self.topo.get_num_hosts()
        min_bw = 10000.0 / float(self.topo.conf["max_capacity"])
        action_min = np.empty(num_actions)
        action_min.fill(min_bw)
        action_max = np.empty(num_actions)
        action_max.fill(1.0)
        self.action_space = spaces.Box(
            low=action_min, high=action_max, dtype=np.float32)
        # Initialize the action arrays shared with the control manager
        # Qdisc do not go beyond uint32 rate limit which is about 4Gbps
        tx_rate = RawArray(ctypes.c_uint32, num_actions)
        self.tx_rate = dc_utils.shmem_to_nparray(tx_rate, np.float32)
        active_rate = RawArray(ctypes.c_uint32, num_actions)
        self.active_rate = dc_utils.shmem_to_nparray(active_rate, np.float32)
        # log.info("%s Setting action space", (self.short_id))
        # log.info("from %s", action_min)
        # log.info("to %s", action_max)
        log.debug(f"{self.short_id} Setting action space from {action_min} to {action_max}")
        # set the observation space
        num_ports = self.topo.get_num_sw_ports()
        num_features = len(self.conf["state_model"])
        if self.conf["collect_flows"]:
            num_features += num_actions * 2
        obs_min = np.empty(num_ports * num_features) # + num_actions) # Omitted to remove active rate from observation
        obs_min.fill(-np.inf)
        obs_max = np.empty(num_ports * num_features) # + num_actions)
        obs_max.fill(np.inf)
        self.observation_space = spaces.Box(
            low=obs_min, high=obs_max, dtype=np.float64)

    def _set_traffic_matrix(self, index, input_dir, topo):
        traffic_file = topo.get_traffic_pattern(index)
        self.input_file = f"{input_dir}/{topo.get_name()}/{traffic_file}"

    def _start_managers(self):
        # actually generate a topology if it does not exist yet
        if not self.net_man:
            log.debug("%s Starting network manager...", self.short_id)
            self.net_man = NetworkManager(self.topo,
                                          self.conf["agent"].lower())
        # in a similar way start a traffic generator
        if not self.traffic_gen:
            log.debug("%s Starting traffic generator...", self.short_id)
            self.traffic_gen = TrafficGen(self.net_man,
                                          self.conf["transport"],
                                          self.conf["output_dir"])
        # Init the state manager
        if not self.state_man:
            self.state_man = StateManager(self.conf,
                                          self.net_man,
                                          self.conf["stats_dict"])
        # Init the state sampler
        if not self.sampler:
            stats = self.state_man.get_stats()
            self.sampler = StatsSampler(stats, self.tx_rate,
                                        self.reward, self.conf["output_dir"])
            self.sampler.start()
        # the bandwidth controller is reinitialized with every new network
        if not self.bw_ctrl:
            host_map = self.net_man.host_ctrl_map
            self.bw_ctrl = BandwidthController(
                host_map, self.tx_rate, self.active_rate, self.topo.max_bps)
            self.bw_ctrl.start()

    def _start_env(self):
        log.debug("%s Starting environment...", self.short_id)
        # Launch all managers (if they are not active already)
        # This lazy initialization ensures that the environment object can be
        # created without initializing the virtual network
        self._start_managers()
        # Finally, start the traffic
        self.traffic_gen.start(self.input_file)

    def _stop_env(self):
        log.debug("%s Stopping environment...", self.short_id)
        if self.traffic_gen:
            log.debug("%s Stopping traffic", self.short_id)
            self.traffic_gen.stop()
        log.debug("%s Done with stopping.", self.short_id)

    def reset(self, *, seed=None, options=None):
        if self.state_man:
           log.info(f"Resetting environment {self.short_id}, observation {self.state_man.observe()}")
        self._stop_env()
        self._start_env()
        log.debug("%s Done with resetting.", self.short_id)
        return np.zeros(self.observation_space.shape), {}

    def close(self):
        if self.terminated:
            return
        self.terminated = True
        log.debug("%s Closing environment...", self.short_id)
        if self.state_man:
            log.debug("%s Stopping all state collectors...", self.short_id)
            self.state_man.close()
            self.state_man = None
        if self.bw_ctrl:
            log.debug("%s Shutting down bandwidth control...", self.short_id)
            self.bw_ctrl.close()
            self.bw_ctrl = None
        if self.sampler:
            log.debug("%s Shutting down data sampling.", self.short_id)
            self.sampler.close()
            self.sampler = None
        if self.traffic_gen:
            log.debug("%s Shutting down generators...", self.short_id)
            self.traffic_gen.close()
            self.traffic_gen = None
        if self.net_man:
            log.debug("%s Stopping network.", self.short_id)
            self.net_man.stop_network()
            self.net_man = None
        log.debug("%s Done with destroying myself.", self.short_id)

    def step(self, action):
        # Truncate actions to legal values
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # Retrieve observation and reward
        obs = self.state_man.observe()
        obs = np.array(obs)
        self.reward.value = self.state_man.get_reward(action)
        # Retrieve the bandwidth enforced by bandwidth control
        # obs = np.append(obs, self.active_rate)
        # FIX: Move the active rate to info dict so that it is not part of the observation
        info = {}
        info["active_rate"] = self.active_rate
        info["action"] = action # Monitor the action taken & compare with active rate
        
        # Update the array with the bandwidth control
        self.tx_rate[:] = action
        # The environment is finished when the traffic generators have stopped
        terminated = done = not self.traffic_gen.check_if_traffic_alive()
        truncated = False
        return obs, self.reward.value, done, truncated, info

    def _handle_interrupt(self, signum, frame):
        log.warning("%s \nEnvironment: Caught interrupt", self.short_id)
        atexit.unregister(self.close())
        self.close()
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="results/EnvRun2")
    parser.add_argument("--topo", default="dumbbell")
    parser.add_argument("--state_model", default=["backlog"], nargs="+", help="State model (backlog, drops, olimit, bw_rx, bw_tx)")
    parser.add_argument("--reward_model", default=["joint_queue"], nargs="+", help="Reward model (step, std_dev, joint_queue, fair_queue)")
    args = parser.parse_args()
    env = DCEnv(vars(args))
    # env = DCEnv({"output_dir": "results/EnvRun2"})
    print(f"Env config: {env.conf}")
    print("Running environment test from main")

    STEPS = 10
    env.reset()
    
    #Use stable_baselines3 environment check
    from stable_baselines3.common.env_checker import check_env
    import time
    print("Checking environment using stable_baselines3")
    check_env(env, warn=True, skip_render_check=True)
    print("Done checking environment")
    bw = []
    done = False
    # Set the precision of the array to 2 decimal places
    # np.set_printoptions(precision=2)
    print(f"Running environment for {STEPS} steps - Random actions")
    for i in range(STEPS):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        wait = 0.1
        time.sleep(wait)
        
        if env.conf["collect_flows"]:
            pass
            # Format observation to be more readable
        done = terminated or truncated
        print(f"{i+1}. Action: {action}\n Obs: {obs} <- {len(obs)=} \nReward: {reward}, Done: {done}")
        # print(f"Active rate: {info['active_rate']} \nTx rate:    {info['action']}")
        if np.allclose(info['active_rate'], info['action']):
            print("Active rate == Tx rate")
        else:
            print("Active rate != Tx rate")
        print("\n")

    env.reset()
    done = False
    print(f"Running environment for {STEPS} steps - MAX actions")
    for i in range(STEPS):
        action = env.action_space.high
        obs, reward, terminated, truncated, info = env.step(action)
        wait = 0.1
        time.sleep(wait)
        
        if env.conf["collect_flows"]:
            pass
            # Format observation to be more readable
        done = terminated or truncated
        print(f"{i+1}. Action: {action}\n Obs: {obs} <- {len(obs)=} \nReward: {reward}, Done: {done}")
        # print(f"Active rate: {info['active_rate']} \nTx rate:    {info['action']}")
        if np.allclose(info['active_rate'], info['action']):
            print("Active rate == Tx rate")
        else:
            print("Active rate != Tx rate")
        print("\n")

    env.reset()
    done = False
    print(f"Running environment for {STEPS} steps - MIN actions")
    for i in range(STEPS):
        action = env.action_space.low
        obs, reward, terminated, truncated, info = env.step(action)
        wait = 0.1
        time.sleep(wait)
        
        if env.conf["collect_flows"]:
            pass
            # Format observation to be more readable
        done = terminated or truncated
        print(f"{i+1}. Action: {action}\n Obs: {obs} <- {len(obs)=} \nReward: {reward}, Done: {done}")
        # print(f"Active rate: {info['active_rate']} \nTx rate:    {info['action']}")
        if np.allclose(info['active_rate'], info['action']):
            print("Active rate == Tx rate")
        else:
            print("Active rate != Tx rate")
        print("\n")
    env.close()
