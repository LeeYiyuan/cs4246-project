import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
import collections
import random
from env import construct_task2_env
from dqn_agent import DQNAgent
from models import ConvDQN
    
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

DQN_MODEL_PATH = 'dqn.model'
SAVE_INTERVAL = 100
SAVE_PATH = 'dqn_epsilon.model'

learning_rate = 0.000001
gamma         = 0.98
buffer_limit  = 20000
batch_size    = 256
max_episodes  = 10000000
t_max         = 600
min_buffer    = 10000
target_update = 10 # episode(s)
train_steps   = 10
epsilon = 0.20
print_interval= 100

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = []
    
    def push(self, transition):
        self.buffer = (self.buffer + [transition])[-self.buffer_limit:] 
    
    def extend(self, transitions):
        self.buffer = (self.buffer + transitions)[-self.buffer_limit:]
    
    def sample(self, batch_size):
        return random.choices(self.buffer, k=batch_size)

    def __len__(self):
        return len(self.buffer)

def compute_loss(model, target, states, actions, rewards, next_states, dones):
    model_values = model(states).gather(1, actions).squeeze()
    with torch.no_grad():
        target_values = target(next_states).max(1)[0]
        target_values -= torch.mul(dones.squeeze(dim=1), target_values)
        target_values = rewards.squeeze(dim=1) + gamma * target_values
    loss = F.smooth_l1_loss(model_values, target_values)
    return loss

def optimize(model, target, memory, forced_transitions, optimizer):
    sample =  memory.sample(batch_size - len(forced_transitions)) + forced_transitions
    loss = compute_loss(model, target, *(torch.stack([s[i] for s in sample]) for i in range(5)))
    
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss

def get_agent_pos(state):
    for lane in range(10):
        for x in range(50):
            if state[1][lane][x] > 0:
                return (lane, x)

if __name__ == '__main__':
    print('Initializing device and model...')
    model = ConvDQN().to(device)
    model.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=lambda storage, loc: storage))
    target = ConvDQN().to(device)
    target.load_state_dict(torch.load(DQN_MODEL_PATH, map_location=lambda storage, loc: storage))
    target.eval()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print('Initializing environment...')
    env = construct_task2_env() 
    env.reset()
    memory = ReplayBuffer()
    
    print('Training...')

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_rewards = 0.0
        episode_transitions = []

        for t in range(t_max):
            # Model takes action
            if np.random.random() < epsilon:
                action = int(np.random.choice([0, 1, 2, 3, 4]))
            else:
                with torch.no_grad():
                    Q = model(torch.stack([torch.Tensor(state).to(device)]))[0].cpu().numpy()
                    action = int(np.argmax(Q))

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            episode_transitions.append(Transition(
                torch.tensor(state).to(device).float(), 
                torch.tensor(np.array([action])).to(device).long(), 
                torch.tensor(np.array([reward])).to(device).float(),
                torch.tensor(next_state).to(device).float(), 
                torch.tensor(np.array([done])).to(device).float()))

            state = next_state
            episode_rewards += reward
            if done:
                break
        rewards.append(episode_rewards)
        
        # Train the model if memory is sufficient
        if len(memory) + len(episode_transitions) >= min_buffer:
            for i in range(train_steps):
                loss = optimize(model, target, memory, episode_transitions, optimizer)
                losses.append(loss.item())
       
        memory.extend(episode_transitions)

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())
            target.eval()

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}] | avg rewards : {:.3f} | s.d. rewards: {:.3f} | avg loss : {:.10f} | buffer size : {} | epsilon : {:.1f}%".format(
                            episode, np.mean(rewards), np.std(rewards), np.mean(losses), len(memory), epsilon*100))
            rewards = []
            losses = []

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            torch.save(model.state_dict(), SAVE_PATH)

