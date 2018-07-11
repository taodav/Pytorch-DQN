import os
import torch
import gym
from utils.helpers import copy_model_params, device
from train import train, VALID_ACTIONS
from estimator import Estimator

num_episodes = 1000
frame_history_len = 4

# env = gym.envs.make("Breakout-v0")
env = gym.envs.make("BreakoutDeterministic-v4")
if len(env.observation_space.shape) == 1:
    # This means we are running on low-dimensional observations (e.g. RAM)
    input_arg = env.observation_space.shape[0]
else:
    img_h, img_w, img_c = env.observation_space.shape
    img_c = 1  # FOR GRAYSCALE TRAINING
    input_arg = frame_history_len * img_c

num_actions = env.action_space.n
estimator = Estimator(input_arg, VALID_ACTIONS).to(device)
target_network = Estimator(input_arg, VALID_ACTIONS).to(device)
model_path = "./checkpoints/checkpoint.pt"

if os.path.isfile(model_path):
    print("Loading previous weights")
    weights = torch.load(model_path)
    estimator.load_state_dict(weights)
    copy_model_params(estimator, target_network)



train(env, estimator, target_network, num_episodes=num_episodes, frame_history_len=frame_history_len)



