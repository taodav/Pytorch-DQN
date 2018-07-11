import torch
import torch.nn as nn
import torch.nn.functional as F


class Estimator(nn.Module):
    def __init__(self, in_channels, valid_actions):
        """
        Value estimator for DQN
        :param in_channels: number of in channels for Conv2d
        :param valid_actions: all valid actions
        """
        super(Estimator, self).__init__()
        # input shape batch_size x in_channels x 84 x 84
        self.in_channels = in_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.r1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.r2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.r3 = nn.ReLU()

        self.dense = torch.nn.Linear(64 * 8 * 8, 512)
        self.r4 = nn.ReLU()
        self.out = torch.nn.Linear(512, len(valid_actions))

    def forward(self, x):
        """
        Calculates probability of each action
        NOTE: a single discrete state is collection of 4 frames
        :param x: processed state of shape b x in_channel x 84 x 84
        :returns tensor of shape [batch_size, NUM_VALID_ACTIONS] (estimated action values)
        """
        x = self.r1(self.conv1(x))  # b x 32 x 21 x 21

        x = self.r2(self.conv2(x))  # b x 64 x 12 x 12

        x = self.r3(self.conv3(x))

        x = x.view(x.size(0), -1)  # b x (64 * 12 * 12)
        dense_out = self.dense(x)  # b x 512
        dense_out = self.r4(dense_out)
        output = self.out(dense_out)  # b x VALID_ACTIONS
        # gather valid action values for each batch based on a.
        return output
