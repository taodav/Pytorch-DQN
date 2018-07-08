import random
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