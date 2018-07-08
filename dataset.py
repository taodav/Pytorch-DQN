from torch.utils.data import Dataset
import torch
from utils.helpers import process_state

class BreakoutDataset(Dataset):
    def __init__(self, env, state_per=4, max_steps=100):
        """
        Env wrapper that pools 4 steps into one state
        :param env:
        :param state_per:
        """
        super(BreakoutDataset, self).__init__()
        self.env = env
        self.max_steps = max_steps
        state = env.reset()
        # init = torch.tensor([state] * state_per, device=device)
        self.current_state = torch.stack()

    def __len__(self):
        return self.max_steps

    def __getitem__(self, index):
        pass


