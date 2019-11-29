try:
    from runner.abstracts import Agent
except:
    class Agent(object): pass
import torch
import numpy as np
from models import RTrailNetwork

RTRAILNET1_MODEL_PATH = 'rtrailnet1.model'
RTRAILNET2_MODEL_PATH = 'rtrailnet2.model'

def get_agent_pos(state):
    for lane in range(10):
        for x in range(50):
            if state[1][lane][x] > 0:
                return (lane, x)

class ExpertAgent(Agent):
    def __init__(self, *args, **kwargs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.rtrailnet1 = RTrailNetwork()
        self.rtrailnet1.load_state_dict(torch.load(RTRAILNET1_MODEL_PATH, map_location=lambda storage, loc: storage))
        self.rtrailnet1.to(self.device)
        self.rtrailnet1.eval()
        self.rtrailnet2 = RTrailNetwork()
        self.rtrailnet2.to(self.device)
        self.rtrailnet2.load_state_dict(torch.load(RTRAILNET2_MODEL_PATH, map_location=lambda storage, loc: storage))
        self.rtrailnet2.eval()

    def initialize(self, **kwargs):
        self.hidden1 = torch.zeros(10, 50).to(self.device)
        self.hidden2 = torch.zeros(10, 50).to(self.device)

    def step(self, state, *args, **kwargs):
        # Find agent positions.
        (agent_lane, agent_x) = get_agent_pos(state)

        # Always choose Forward[-1] for first step since trail is empty.
        if agent_lane == 9 and agent_x == 49:
            return 4
        
        # Calculate distributions
        with torch.no_grad():
            (next_trail1, self.hidden1) = self.rtrailnet1(torch.cat(
                    [torch.Tensor(state[0]).to(self.device), torch.Tensor(state[3]).to(self.device), self.hidden1], 
                    dim=1))
            (next_trail2, self.hidden2) = self.rtrailnet1(torch.cat(
                    [torch.Tensor(state[0]).to(self.device), torch.Tensor(state[3]).to(self.device), self.hidden2], dim=1))
        
        # Calculate max speed.
        max_speed = -1
        if agent_x - 1 >= 0 and state[0][agent_lane][agent_x - 1] == 0:
            max_speed = -2
            if agent_x - 2 >= 0 and state[0][agent_lane][agent_x - 2] == 0:
                max_speed = -3
        
        # Check if at top row.
        if agent_lane == 0:
            return 5 + max_speed

        # If at or past death line.
        if agent_x <= agent_lane:
            return 0

        # Limit in case agent breaks past death line.
        if agent_x == agent_lane + 1 and max_speed < -1:
            max_speed = -1

        # Limit in case agent breaks past death line.
        if agent_x == agent_lane + 2 and max_speed < -2:
            max_speed = -2
       
        # Setup probability tables.
        clear_probabilities = dict()
        clear_probabilities[0] = 0.0
        for speed in range(-1, max_speed - 1, -1):
            clear_probabilities[speed] = 0.0

        # Calculate probabilities for immediate up.
        if agent_x - 1 >= 0:
            clear_probabilities[0] = 1 - next_trail1[agent_lane - 1][agent_x - 1].item()

        # Calculate probabilities for 1-delayed up.
        for speed in range(-1, max_speed - 1, -1):
            if agent_x - 1 + speed >= 0:
                clear_probabilities[speed] = 1 - next_trail2[agent_lane - 1][agent_x - 1 + speed].item()
             # If 1-delayed up goes past grid, probability becomes 0, so that the speed 
             # will not be selected.

        # Determine best action.
        if clear_probabilities[0] >= 0.95:
            return 0
        for speed in range(-1, max_speed - 1, -1):
            if clear_probabilities[speed] >= 0.98:
                return 5 + speed
        return 4 
    
    def update(self, *args, **kwargs):
        pass
