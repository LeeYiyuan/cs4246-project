import sys
import time
from env import construct_task2_env
import numpy as np
from models import RTrailNetwork
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAIL_OFFSET = int(sys.argv[1])
SAVE_INTERVAL = 100
SAVE_PATH = 'rtrailnet{}.model'.format(TRAIL_OFFSET)

learning_rate = 0.001

def get_cars(state):
    return torch.Tensor(state[0][0:9])

def get_trails(state):
    return torch.Tensor(state[3][0:9])

if __name__ == '__main__':
    print('Initializing device and model...')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = RTrailNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Initializing environment...')
    env = construct_task2_env() 
    env.reset()
    history = []

    print('Training...')

    input_cartrails = []
    target_trails = []

    iterations = 0
    while True:
        next_state, reward, done, info = env.step(4)
        if not done:
            history.append((
                get_cars(next_state).to(device), 
                get_trails(next_state).to(device)))
            continue

        # Initial hidden variables.
        hidden = torch.Tensor([[0.0] * 50] * 9).to(device)
        
        losses = []

        for i in range(0, len(history) - TRAIL_OFFSET):
            # Belief + hidden after first car and trail.
            (belief, hidden) = model(torch.cat(
                [history[i][0], history[i][1], hidden], 
                dim=1))

            # Calculate loss with trial i + TRAIL_OFFSET.
            loss = F.binary_cross_entropy(belief, history[i + TRAIL_OFFSET][1])
            losses.append(loss)

        if len(losses) > 0:
            optimizer.zero_grad()
            loss = sum(losses) / len(losses)
            loss.backward()
            optimizer.step()

            print(iterations, loss)
        
            iterations += 1

            if iterations % SAVE_INTERVAL == 0:
                torch.save(model.state_dict(), SAVE_PATH)

        env.reset()
        history = []
