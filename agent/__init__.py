import random
import torch
import numpy as np
from dqn_agent import DQNAgent
from dqn_pair_agent import DQNPairAgent
from expert_agent import ExpertAgent
from actor_agent import ActorAgent

def create_agent(test_case_id, *args, **kwargs):
    return DQNAgent(model_path='model.pt', device='cuda:1')

def get_agent_pos(state):
    for lane in range(10):
        for x in range(50):
            if state[1][lane][x] > 0:
                return (lane, x)

if __name__ == '__main__':
    import sys
    import time
    from env import construct_task1_env, construct_task2_env

    FAST_DOWNWARD_PATH = "/fast_downward/"

    def test(agent, env, runs=1000, t_max=100):
        rewards = []
        fail = []
        for run in range(runs):
            env = construct_task2_env(random_seed=run)
            state = env.reset()
            agent_init = {'fast_downward_path': FAST_DOWNWARD_PATH, 'agent_speed_range': (-3,-1), 'gamma' : 1}
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
                (agent_lane, agent_x) = get_agent_pos(next_state)
                state = next_state
                episode_rewards += reward
                if done:
                    break
            if episode_rewards == 0:
                fail.append(run)
            rewards.append(episode_rewards)
            print(run, episode_rewards, t)
        avg_rewards = sum(rewards)/len(rewards)

        print("{} run(s) avg rewards : {:.3f}".format(runs, avg_rewards))
        print("Fail: " + str(fail))
        return avg_rewards

    def timed_test(task):
        start_time = time.time()
        rewards = []
        for tc in task['testcases']:
            agent = create_agent(tc['id']) # `test_case_id` is unique between the two task 
            print("[{}]".format(tc['id']), end=' ')
            avg_rewards = test(agent, tc['env'], tc['runs'], tc['t_max'])
            rewards.append(avg_rewards)
        point = sum(rewards)/len(rewards)
        elapsed_time = time.time() - start_time

        print('Point:', point)

        for t, remarks in [(0.4, 'fast'), (0.6, 'safe'), (0.8, 'dangerous'), (1.0, 'time limit exceeded')]:
            if elapsed_time < task['time_limit'] * t:
                print("Local runtime: {} seconds --- {}".format(elapsed_time, remarks))
                print("WARNING: do note that this might not reflect the runtime on the server.")
                break

    def get_task(task_id):
        if task_id == 1:
            test_case_id = 'task1_test'
            return { 
                'time_limit' : 600,
                'testcases' : [{'id' : test_case_id, 'env' : construct_task1_env(), 'runs' : 1, 't_max' : 50}]
                }
        elif task_id == 2:
            tcs = [('t2_tmax50', 50), ('t2_tmax40', 40)]
            return {
                'time_limit': 600,
                'testcases': [{ 'id': tc, 'env': construct_task2_env(), 'runs': 1000, 't_max': t_max } for tc, t_max in tcs]
            }
        else:
            raise NotImplementedError

    try:
        task_id = int(sys.argv[1])
    except:
        print('Run agent on an example task.')
        print('Usage: python __init__.py <task number>')
        print('Example:\n   python __init__.py 2')
        exit()

    print('Testing on Task {}'.format(task_id))

    task = get_task(task_id)
    timed_test(task)
