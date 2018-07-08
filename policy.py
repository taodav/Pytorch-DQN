import numpy as np
import torch
from utils.helpers import process_state

def make_epsilon_greedy_policy(estimator, nA):
    """

    :param estimator: model that returns q values for a given statem/action pair
    :param nA: number of actions in the environment
    :return: A function that takes in a state and an epsilon and returns probs for each
        action in the form of a numpy array of length nA
    """
    def policy_fn(state, epsilon):
        """
        :param state: tensor of b x 1 x 84 x 84
        :param epsilon:
        :return: action probabilities, of size b x nA
        """
        A = torch.ones((state.size(0), nA)) * epsilon / nA
        q_vals = estimator.forward(state)
        best_action = torch.argmax(q_vals, dim=1).unsqueeze(-1)  # b
        A[:, best_action] += (1.0 - epsilon)
        return A
    return policy_fn