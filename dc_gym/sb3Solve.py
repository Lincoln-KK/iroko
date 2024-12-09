
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
from dataclasses import dataclass, field
import tyro
from typing import Callable, List

import wandb
from wandb.integration.sb3 import WandbCallback
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym
from gymnasium.envs.registration import register
import numpy as np
import os
import logging
from datetime import datetime
# parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.sys.path.insert(0, parent_dir)

log = logging.getLogger(__name__)
# log = get_logger(__name__)

ENV_NAME = "dc-iroko-v0"


#Use Tyro for CLI arguments
@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    wandb_project_name: str = "sb3_iroko_state"
    """the name of the wandb project"""
    # Algorithm specific arguments
    env_name: str = ENV_NAME
    """the id of the environment"""
    num_envs: int = 1
    """the number of parallel game environments"""
    seed: int = 0
    """the seed for the random number generator"""
    batch_size: int = 1024
    """size of the batch"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    gamma: float = 0.99
    """the discount factor gamma"""
    device: str = "auto"
    """The device to use for training"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = True
    """if toggled, this experiment will not be tracked with Weights and Biases"""
    save_model: bool = True
    """whether to save model into the `runs/{run_name}` folder"""

    # Environment specific arguments
    max_steps: int = 40000
    """The maximum number of steps to take in the environment"""
    reward_model: str = "default"
    """Reward function to use (e.g. joint, step, fair_queue)"""
    total_timesteps: int = 100000 
    """total timesteps of the experiments"""
    state_model: List[str] = "drops" #field(default_factory=lambda: ["drops"])
    """the state model to use (backlog, drops, olimit, bw_rx, bw_tx)"""
    reward_model: List[str] = "fairness"
    """the reward model to use (fairness, gini, action, fair_queue, joint_queue, step)"""
    policy_type: str = "MlpPolicy"
    """the policy type to use (MlpPolicy, CnnPolicy)"""


args = tyro.cli(Args)
config = vars(args)
config["exp_name"] = args.exp_name + "s_" + ",".join(args.state_model)



register(id="dc-iroko-v0", entry_point="env_iroko:DCEnv")
# env = gym.make(config["env_name"])

def make_env():
    try:
        env = gym.make(config["env_name"], conf=config)
    except TypeError as e:
        log.warning(f"Environment {config['env_name']} does not support 'conf' argument: {e}")
        env = gym.make(config["env_name"])
    env = Monitor(env)
    # env = gym.wrappers.RecordEpisodeStatistics(env)
    # env = gym.wrappers.TimeLimit(env, max_episode_steps=config["max_steps"])
    env = gym.wrappers.NormalizeReward(env, gamma=args.gamma) # Handles reward discounting
    return env

# env = DummyVecEnv([make_env])
env = make_env()
try:
    print(env.unwrapped.conf)
except AttributeError:
    log.warning("The environment does not have a 'conf' attribute.")
check_env(env)

# # Passes. Recommends symmetric and normalized Box action space (range=[-1, 1])

if args.track:
    # Initialize wandb
    run=wandb.init(project=args.wandb_project_name, config=config, 
               sync_tensorboard=True, save_code=True)


# Use PPO to learn a policy & evaluate it

model = PPO(config["policy_type"], env, 
            seed=config["seed"],
            learning_rate=config["learning_rate"],
            gamma=config["gamma"],
            batch_size=config["batch_size"], verbose=1, 
            tensorboard_log="logs/SB3_iroko_states", device=args.device) 
TIMESTEPS = config["total_timesteps"]
N_EVAL_EPISODES = 10

print(f"Training for {TIMESTEPS} timesteps starting now.... Wandb tracking: {args.track}")
if args.track:
    model.learn(total_timesteps=config["total_timesteps"],
        tb_log_name=config["exp_name"]+"_"+datetime.now().strftime("%Y%m%d-%H%M%S"),
        progress_bar=True,
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}",
            verbose=2
        ))
else:
    model.learn(total_timesteps=config["total_timesteps"],
        tb_log_name=config["exp_name"]+"_"+datetime.now().strftime("%Y%m%d-%H%M%S"),
        progress_bar=True,

    )

# Save the model
modelname = "".join(m for m in args.state_model)
if args.save_model:
    model.save(f"models/{modelname}_{run.id}")
print(f"Model saved as models/{modelname}_{run.id}")
# Evaluate the model
# N_episodes = 10
# rewards, lengths = evaluate_policy(model, env, return_episode_rewards=True, n_eval_episodes=N_episodes)
# log.debug(f"**After training: rewards={rewards}, lengths={lengths}")
# log.debug(f" mean_reward={np.mean(rewards)} +/- {np.std(rewards)} over {N_episodes} episodes")

print("Training done!!")
# Close the environment
env.close()
run.finish()

 

