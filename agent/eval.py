import glob
import os
import random
import sys
import time
import multiprocessing as mp
from queue import Empty

import torch
import numpy as np

from env import construct_task2_env
from actor_agent import ActorAgent
from dqn_agent import DQNAgent
from expert_agent import ExpertAgent

def create_agent(*args, **kwargs):
    return DQNAgent(*args, **kwargs)

def score(device, jobs, results):
    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        for run in range(runs):
            env = construct_task2_env(run)
            state = env.reset()
            agent_init = {'agent_speed_range': (-3,-1), 'gamma' : 1}
            agent.initialize(**agent_init)
            episode_rewards = 0.0
            for t in range(t_max):
                action = agent.step(state)   
                next_state, reward, done, info = env.step(action)
                full_state = {
                    'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 
                    'done': done, 'info': info
                }
                agent.update(**full_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            rewards.append(episode_rewards)
        avg_rewards = sum(rewards)/len(rewards)
        
        return avg_rewards

    def timed_test(task, agent):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time
        return rewards

    def get_task():
        tcs = [('t2_tmax40', 40)]
        return {
            'time_limit': 600,
            'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 1000, 't_max': t_max } for tc, t_max in tcs]
        }

    while True:
        try:
            job = jobs.get(False)
        except Empty:
            time.sleep(5)
            continue

        task = get_task()
        rewards = timed_test(task, create_agent(model_path=job, device=device))
        results.put((job, rewards))
        print('{} : {}\t| Rem.: {} | Com.: {}'.format(job, rewards, jobs.qsize(), results.qsize()))

if __name__ == '__main__':
    jobs = mp.Queue()
    results = mp.Queue()
    processed = set()

    def refresh_queue():
        list_of_files = glob.glob('trained6/dqn_active.model.*')
        list_of_files = [(f, int(f.split('trained6/dqn_active.model.')[1])) for f in list_of_files]
        list_of_files = [x for x in list_of_files if x[1] not in processed]
        list_of_files.sort(key=lambda x: x[1])
        processed.update([x[1] for x in list_of_files])
        for x in list_of_files:
            jobs.put(x[0])

    processes = []
    for i in range(12):
        p = mp.Process(target=score, args=('cuda:{}'.format([0, 2, 3][i % 3]), jobs, results))
        p.start()
        processes.append(p)
            
    while True:
        refresh_queue()
        time.sleep(10)
