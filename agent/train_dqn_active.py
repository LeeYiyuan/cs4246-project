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

DQN_MODEL_PATH = 'dqn_epsilon.model'
SAVE_INTERVAL = 100
SAVE_PATH = 'trained6/dqn_active.model.{}'

learning_rate = 0.0000001
gamma         = 0.98
buffer_limit  = 10000
batch_size    = 256
max_episodes  = 1000000000
t_max         = 600
min_buffer    = 100
target_update = 10 # episode(s)
train_steps   = 10
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

def optimize(model, target, memory_success, memory_failure, forced_transitions, optimizer):
    sample_success = memory_success.sample(int(batch_size * len(memory_success) / (len(memory_success) + len(memory_failure))))
    sample_failure = memory_failure.sample(int(batch_size * len(memory_failure) / (len(memory_success) + len(memory_failure))))
    sample_forced = forced_transitions
    loss = compute_loss(model, target, *(torch.stack([s[i] for s in sample_success]).to(device) for i in range(5)))
    loss += compute_loss(model, target, *(torch.stack([s[i] for s in sample_failure]).to(device) for i in range(5)))
    loss += compute_loss(model, target, *(torch.stack([s[i] for s in sample_forced]).to(device) for i in range(5))) / train_steps

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
    memory_success = ReplayBuffer()
    memory_failure = ReplayBuffer()
    
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
        if len(memory_success) >= min_buffer and len(memory_failure) >= min_buffer:
            for i in range(train_steps):
                loss = optimize(model, target, memory_success, memory_failure, episode_transitions, optimizer)
                losses.append(loss.item())

            # Update target network every once in a while
            if episode % target_update == 0:
                target.load_state_dict(model.state_dict())
                target.eval()
        
        if episode_rewards > 0:
            memory_success.extend(episode_transitions)
        else :
            memory_failure.extend(episode_transitions)


        if episode % print_interval == 0 and episode > 0:
            print("[Episode {}] | avg rewards : {:.3f} | s.d. rewards: {:.3f} | avg loss : {:.10f} | succ. buffer : {} | fail. buffer : {}".format(
                            episode, np.mean(rewards), np.std(rewards), np.mean(losses), len(memory_success), len(memory_failure)))
            rewards = []
            losses = []

        if episode % SAVE_INTERVAL == 0 and episode > 0:
            torch.save(model.state_dict(), SAVE_PATH.format(episode))

