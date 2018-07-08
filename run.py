import os
import torch
import gym
from utils.helpers import copy_model_params, device
from train import train, VALID_ACTIONS
from estimator import Estimator

num_episodes = 1000
in_channels = 3
estimator = Estimator(in_channels, VALID_ACTIONS).to(device)
target_network = Estimator(in_channels, VALID_ACTIONS).to(device)
model_path = "./checkpoints/checkpoint.pt"

if os.path.isfile(model_path):
    print("Loading previous weights")
    weights = torch.load(model_path)
    estimator.load_state_dict(weights)
    copy_model_params(estimator, target_network)


env = gym.envs.make("Breakout-v0")

train(env, estimator, target_network, num_episodes=num_episodes)



