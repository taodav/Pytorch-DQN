import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import itertools
from utils.helpers import device, copy_model_params, process_state
from policy import make_epsilon_greedy_policy
from replay import ReplayBuffer

import gym
from gym.wrappers import Monitor

# we should learn this ourselves.
VALID_ACTIONS = [0, 1, 2, 3]

def train(env, estimator, target_network, num_episodes=1000,
                    replay_memory_size=500000,
                    frame_history_len=4,
                    save_every=10,
                    update_every=1000,
                    discount=0.99, epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=50000,
                    batch_size=32, record_every=50):
    """
    deep q learning algorithm
    :param env: openAI gym environment
    :param estimator: estimator model for predicting values
    :param target_network:
    :param num_episodes: number of episodes to run
    :param replay_memory_size: size of replay memory
    :param update_every: copy params from estimator into target estimator after this many steps
    :param discount: discount factor
    :param epsilon_start: starting epsilon value
    :param epsilon_end: ending epsilon value
    :param batch_size: 32 lol
    :param record_every: record a video every N episodes
    :return:
    """

    # Load previous state here
    replay_memory = ReplayBuffer(replay_memory_size, frame_history_len)

    # epsilon delay schedule
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    loss_func = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(estimator.parameters())

    policy = make_epsilon_greedy_policy(estimator, len(VALID_ACTIONS))

    env = Monitor(env, directory="./monitor",
                  resume=True,
                  video_callable=lambda count: count % record_every == 0)

    total_t = 0
    pbar = tqdm(range(num_episodes))
    pbar.set_description("ep: %d, er: %.2f, et: %d, tt: %d, exp_size: %d" % (0, 0.0, 0, 0, 0))

    for ep in pbar:

        state = env.reset()  # 210 x 160 x 4
        state = process_state(state)  # 94 x 94 x 3
        episode_loss = 0
        episode_reward = 0
        episode_t = 0

        for t in itertools.count():
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            last_idx = replay_memory.store_frame(state)

            recent_observations = replay_memory.encode_recent_observation()

            action_dist = policy(recent_observations, epsilon)
            action_dist = action_dist.squeeze(0).numpy()
            action = np.random.choice(np.arange(len(action_dist)), p=action_dist)

            next_state, reward, done, _ = env.step(action)
            reward = max(-1.0, min(reward, 1.0))

            episode_reward += reward

            replay_memory.store_effect(last_idx, action, reward, done)
            next_state = process_state(next_state)

            state = next_state

            if replay_memory.can_sample(batch_size):
                obs_batch, act_batch, rew_batch, next_obs_batch, done_mask = replay_memory.sample(batch_size)
                obs_batch = torch.from_numpy(obs_batch).float()
                obs_batch = obs_batch.to(device)
                act_batch = torch.from_numpy(act_batch).long().to(device) / 255.0
                rew_batch = torch.from_numpy(rew_batch).to(device)
                next_obs_batch = torch.from_numpy(next_obs_batch).float().to(device) / 255.0
                not_done_mask = torch.from_numpy(1 - done_mask).float().to(device)

                state_values = estimator(obs_batch)  # b x VALID_ACTIONS
                state_action_values = torch.gather(state_values, 1, act_batch.unsqueeze(1))  # b x 1

                next_state_values_max = target_network(next_obs_batch).detach().max(dim=1)[0]
                next_state_values = not_done_mask * next_state_values_max

                expected_q_value = (next_state_values * discount) + rew_batch

                # bellman_error = expected_q_value - state_action_values.squeeze(1)
                #
                # clipped_bellman_error = bellman_error.clamp(-1, 1)
                #
                # d_error = clipped_bellman_error * -1.0

                loss = loss_func(state_action_values, expected_q_value.unsqueeze(1))
                episode_loss += loss

                # state_action_values.backward(d_error.data.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if done:
                break

            total_t += 1
            episode_t = t

        pbar.set_description("ep: %d, el: %.5f, er: %.2f, et: %d, tt: %d, exp_size: %d" % (ep, episode_loss, episode_reward, episode_t, total_t, replay_memory.num_in_buffer))
        if total_t % update_every == 0:
            copy_model_params(estimator, target_network)

        # save checkpoint
        if ep % save_every == 0:
            torch.save(estimator.state_dict(), './checkpoints/checkpoint.pt')

    env.close()



