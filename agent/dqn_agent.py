try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from models import ConvDQN

class DQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        self.model_path = kwargs['model_path']
        self.device = torch.device(kwargs['device'])
        self.dqn = ConvDQN()
        self.dqn.load_state_dict(torch.load(self.model_path, map_location=lambda storage, loc: storage))
        self.dqn.to(self.device)
        self.dqn.eval()

    def initialize(self, **kwargs):
        pass

    def step(self, state, *args, **kwargs):
        with torch.no_grad():
            Q = self.dqn(torch.stack([torch.Tensor(state).to(self.device)]))[0].cpu().numpy()
            return int(np.argmax(Q))
        
    def update(self, *args, **kwargs):
        pass
