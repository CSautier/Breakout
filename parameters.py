import torch


class Parameters:
    def __init__(self):
        self.ACTOR_COEFF = 1.
        self.LOSS_CLIPPING = 0.15
        self.GAMMA = 0.98
        self.DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.BATCH_SIZE = 32
        self.EPOCH_STEPS = 10
        self.MAXLEN = 1000
        self.LEARNING_RATE = 1e-4


parameters = Parameters()
