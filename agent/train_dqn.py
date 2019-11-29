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
import os
from env import construct_task2_env
from expert_agent import ExpertAgent
from models import ConvDQN
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

SAVE_INTERVAL = 100
SAVE_PATH = 'dqn.model'

# Hyperparameters --- don't change, RL is very sensitive
learning_rate = 0.0001
gamma         = 0.98
buffer_limit  = 20000
batch_size    = 128
max_episodes  = 10000000
t_max         = 600
min_buffer    = 10000
target_update = 10 # episode(s)
train_steps   = 10
max_epsilon   = 1.00
min_epsilon   = 0.00
epsilon_decay = 5000
print_interval= 100

Transition = collections.namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer():
    def __init__(self, buffer_limit=buffer_limit):
        self.buffer_limit = buffer_limit
        self.buffer = []
    
    def push(self, transition):
        self.buffer = (self.buffer + [transition])[-self.buffer_limit:] 
    
    def sample(self, batch_size):
        samples = random.choices(self.buffer, k=batch_size)
        return (
                torch.stack([s[0] for s in samples]),
                torch.stack([s[1] for s in samples]),
                torch.stack([s[2] for s in samples]),
                torch.stack([s[3] for s in samples]),
                torch.stack([s[4] for s in samples]))

    def __len__(self):
        return len(self.buffer)

def compute_epsilon(episode):
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * math.exp(-1. * episode / epsilon_decay)
    return epsilon

def compute_loss(model, target, states, actions, rewards, next_states, dones):
    model_values = model(states).gather(1, actions).squeeze()
    with torch.no_grad():
        target_values = target(next_states).max(1)[0]
        target_values -= torch.mul(dones.squeeze(dim=1), target_values)
        target_values = rewards.squeeze(dim=1) + gamma * target_values
    loss = F.smooth_l1_loss(model_values, target_values)
    return loss

def optimize(model, target, memory, optimizer):
    batch = memory.sample(batch_size)
    loss = compute_loss(model, target, *batch)
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
    print('Initializing model...')
    model = ConvDQN().to(device)
    target = ConvDQN().to(device)

    if os.path.exists(SAVE_PATH):
        print('Resuming training from previous training.')
        model.load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage))
        target.load_state_dict(torch.load(SAVE_PATH, map_location=lambda storage, loc: storage))
    else:
        print('Starting new training.')
    target.eval()
    expert = ExpertAgent()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print('Initializing environment...')
    env = construct_task2_env() 
    env.reset()
    expert.initialize()
    memory = ReplayBuffer()
    
    print('Training...')

    # Initialize rewards, losses, and optimizer
    rewards = []
    losses = []

    for episode in range(max_episodes):
        epsilon = compute_epsilon(episode)
        env.reset()
        expert.initialize()
        state = env.step(4)[0] # Move forward on first step.
        episode_rewards = 0.0

        for t in range(t_max):
            # Model takes action
            if np.random.random() < epsilon:
                action = expert.step(state)
            else:
                with torch.no_grad():
                    action = np.argmax(model(torch.stack([torch.Tensor(state).to(device)]))[0].cpu().numpy())
                    action = int(action)

            # Apply the action to the environment
            next_state, reward, done, info = env.step(action)

            # Save transition to replay buffer
            memory.push(Transition(
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
        if len(memory) > min_buffer:
            for i in range(train_steps):
                loss = optimize(model, target, memory, optimizer)
                losses.append(loss.item())

        # Update target network every once in a while
        if episode % target_update == 0:
            target.load_state_dict(model.state_dict())
            target.eval()

        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}] | avg rewards : {:.3f}, | avg loss : {:.10f} | buffer size : {} | epsilon : {:.1f}%".format(
                            episode, np.mean(rewards), np.mean(losses), len(memory), epsilon*100))
            rewards = []
            losses = []

        if episode % SAVE_INTERVAL == 0:
            torch.save(model.state_dict(), SAVE_PATH)

