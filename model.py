import torch
import torch.nn as nn
import torch.nn.functional as F
from parameters import parameters


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return torch.eye(num_classes)[y].to(parameters.DEVICE)


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.vision = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.MaxPool2d(2, 2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(6, 16, 5),
            nn.MaxPool2d(2),
            nn.LeakyReLU(0.1),
        )

        self.actor = nn.Sequential(
            nn.Linear(16 * 49 * 37, 120),
            nn.LeakyReLU(0.1),
            nn.Linear(120, 84),
            nn.LeakyReLU(0.1),
            nn.Linear(84, 4)
        )

        self.critic = nn.Sequential(
            nn.Linear(16 * 49 * 37, 120),
            nn.LeakyReLU(0.1),
            nn.Linear(120, 84),
            nn.LeakyReLU(0.1),
            nn.Linear(84, 1)
        )

    def forward(self, x):
        vision = self.vision(x).view(-1, 29008)
        actor = self.actor(vision).view(-1, 4).softmax(-1)
        if self.training:
            critic = self.critic(vision)
            return actor, critic.squeeze()
        return actor

    def loss(self, observations, rewards, actions, old_prob):
        prob_distribution, reward_predicted = self.forward(observations)
        r = (torch.sum(torch.eye(4)[actions].to(parameters.DEVICE) * prob_distribution, -1) + 1e-10) / (old_prob + 1e-10)
        advantage = (rewards - reward_predicted).detach()
        lossactor = - parameters.ACTOR_COEFF \
                    * torch.mean(torch.min(r * advantage,
                                           torch.clamp(r,
                                                       min=(1. - parameters.LOSS_CLIPPING),
                                                       max=(1. + parameters.LOSS_CLIPPING))
                                           * advantage))
        losscritic = F.mse_loss(reward_predicted, rewards)
        return lossactor, losscritic
