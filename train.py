import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import itertools
from utils.helpers import device, copy_model_params, process_state
from policy import make_epsilon_greedy_policy
from replay import ReplayMemory

import gym
from gym.wrappers import Monitor

# we should learn this ourselves.
VALID_ACTIONS = [0, 1, 2, 3]

def train(env, estimator, target_network, num_episodes=1000,
                    replay_memory_size=10000,
                    save_every=10,
                    update_every=10000,
                    discount=0.99, epsilon_start=1.0,
                    epsilon_end=0.1,
                    epsilon_decay_steps=10000,
                    batch_size=16, record_every=50):
    """
    deep q learning algorithm
    :param env: openAI gym envrionment
    :param estimator: estimator model for predicting values
    :param target_network:
    :param num_episodes: number of episodes to run
    :param replay_memory_size: size of replay memory
    :param replay_memory_init_size: number of random experiences to sample initally
    :param update_every: copy params from estimator into target estimator after this many steps
    :param discount: discount factor
    :param epsilon_start: starting epsilon value
    :param epsilon_end: ending epsilon value
    :param batch_size: 32 lol
    :param record_every: record a video every N episodes
    :return:
    """

    # Load previous state here
    replay_memory = ReplayMemory(replay_memory_size)

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
    pbar.set_description("ep: %d, el: %.5f, er: %.2f, et: %d, tt: %d, exp_size: %d" % (0, 0.0, 0.0, 0, 0, 0))

    for ep in pbar:

        state = env.reset()
        state = process_state(state)  # 1 x 84 x 84
        state = state.expand(estimator.in_channels, state.size(1), state.size(2))
        episode_loss = 0
        episode_reward = 0
        episode_t = 0

        for t in itertools.count():
            epsilon = epsilons[min(total_t, epsilon_decay_steps - 1)]

            action_dist = policy(state.unsqueeze(0), epsilon)
            action_dist = action_dist.squeeze(0).numpy()
            action = np.random.choice(np.arange(len(action_dist)), p=action_dist)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            next_state = process_state(next_state)
            next_state = torch.cat((state[1:, :, :], next_state))

            replay_memory.push(state.cpu().numpy(), action, next_state, reward)

            if len(replay_memory) < batch_size:
                continue

            transitions = replay_memory.sample(batch_size)
            batch = replay_memory.transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), \
                                          device=device, dtype=torch.uint8)
            non_final_next_state = torch.stack([s for s in batch.next_state if s is not None])

            tensor_state_batch = [torch.tensor(arr, dtype=torch.float, device=device) for arr in batch.state]
            state_batch = torch.stack(tensor_state_batch)  # b x 1 x 84 x 84
            action_batch = torch.tensor(list(batch.action), dtype=torch.long, device=device).unsqueeze(1)  # b x 1
            reward_batch = torch.tensor(list(batch.reward), dtype=torch.float, device=device).unsqueeze(1)  # b x 1

            state_values = estimator(state_batch)  # b x VALID_ACTIONS
            state_action_values = torch.gather(state_values, 1, action_batch)  # b x 1

            next_state_values = torch.zeros(batch_size, device=device).unsqueeze(-1)  # b
            next_state_values_max = target_network(non_final_next_state).max(dim=1)[0]
            next_state_values[non_final_mask] = next_state_values_max.detach().unsqueeze(-1)  # b x 1

            expected_q_value = (next_state_values * discount) + reward_batch

            loss = loss_func(state_action_values, expected_q_value)
            episode_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if done:
                break

            total_t += 1
            state = next_state
            episode_t = t

        pbar.set_description("ep: %d, el: %.5f, er: %.2f, et: %d, tt: %d, exp_size: %d" % (ep, episode_loss, episode_reward, episode_t, total_t, len(replay_memory)))
        if total_t % update_every == 0:
            copy_model_params(estimator, target_network)

        # save checkpoint
        if ep % save_every == 0:
            torch.save(estimator.state_dict(), './checkpoints/checkpoint.pt')

    env.close()



