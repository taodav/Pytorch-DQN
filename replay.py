import random
import numpy as np
from collections import deque, namedtuple
from utils.helpers import process_state, device

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = deque(maxlen=self.capacity)
        self.transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, *args):
        transition = self.transition(*args)
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

def sample_n_unique(sampling_f, n):
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res

class ReplayBuffer(object):
    def __init__(self, size, frame_history_len):
        """
        Memory-efficient replay memory.
        :param size: Max size of replay memory
        :param frame_history_len: how many frames to package into one state
        """
        self.size = size
        self.frame_history_len = frame_history_len

        self.next_idx      = 0
        self.num_in_buffer = 0

        self.obs      = None
        self.action   = None
        self.reward   = None
        self.done     = None

    def can_sample(self, batch_size):
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        """
        Given indexes, encodes the samples in replay buffer
        :param idxes: list of indices
        :return: observation batch, action batch, reward batch, next obs batch, done mask
        """
        obs_batch = np.concatenate([self._encode_observation(idx)[np.newaxis, :] for idx in idxes], 0)
        act_batch = self.action[idxes]
        rew_batch = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[np.newaxis, :] for idx in idxes], 0)
        done_mask = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """
        samples from replay buffer
        :param batch_size: how many transitions to sample
        :return: observations_batch (b x img_c * frame_history_length x img_h x img_w),
         actions_batch (b),
         reward_batch (b),
         next_obs_batch (b x img_c * frame_history_length x img_h x img_w),
         done_mask (b)
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """
        returns most recent `frame_history_len` frames
        :return: np array of shape (frame_history_length * img_c x img_h x img_w
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    def _encode_observation(self, idx):
        """
        returns encoded observation at index idx
        :param idx: index to encode
        :return: array of size (img_c * frame_history_len, img_h, img_w)
        """
        end_idx = idx + 1
        start_idx = end_idx - self.frame_history_len

        if len(self.obs.shape) == 2:
            return self.obs[end_idx - 1]

        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0

        # for idx in range(start_idx, end_idx - 1):
        #     if self.done[idx % self.size]:
        #         print("before", start_idx, end_idx)
        #         start_idx = idx + 1
        #         print("after", start_idx, end_idx)

        missing_context = self.frame_history_len - (end_idx - start_idx)

        if start_idx < 0 or missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 0)
        else:
            img_h, img_w = self.obs.shape[2], self.obs.shape[3]
            return self.obs[start_idx:end_idx].reshape(-1, img_h, img_w)

    def store_frame(self, frame):
        """
        stores a single frame in the buffer
        :param frame: frame of shape (img_h, img_w, img_c)
        :return: index in which frame is stored. Used for storing the rest of transition
        """
        # if len(frame.shape) > 1:
        #     frame = frame.transpose(2, 0, 1)

        if self.obs is None:
            self.obs      = np.empty([self.size] + list(frame.shape), dtype=np.uint8)
            self.action   = np.empty([self.size],                     dtype=np.int32)
            self.reward   = np.empty([self.size],                     dtype=np.float32)
            self.done     = np.empty([self.size],                     dtype=np.bool)

        self.obs[self.next_idx] = frame
        res = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)
        return res

    def store_effect(self, idx, action, reward, done):
        """
        stores all the other stuff.
        :param idx:
        :param action:
        :param reward:
        :param done:
        :return:
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = done