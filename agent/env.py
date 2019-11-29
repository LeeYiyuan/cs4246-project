import gym
from gym_grid_driving.envs.grid_driving import LaneSpec

def construct_task2_env(random_seed=None):
    config = {'observation_type': 'tensor', 'agent_speed_range': [-3, -1], 'width': 50,
              'lanes': [LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -1]), 
                        LaneSpec(cars=7, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -1]), 
                        LaneSpec(cars=6, speed_range=[-3, -2]), 
                        LaneSpec(cars=7, speed_range=[-1, -1]), 
                        LaneSpec(cars=6, speed_range=[-2, -1]), 
                        LaneSpec(cars=8, speed_range=[-2, -2])],
              'random_seed' : random_seed
            }
    return gym.make('GridDriving-v0', **config)

def construct_task1_env() :
    test_config = [{'lanes' : [LaneSpec(10, [-2, -2])] *2 + [LaneSpec(10, [-3, -3])] *2 +
                              [LaneSpec(8, [-4, -4])] *2 + [LaneSpec(8, [-5, -5])] *2 +
                              [LaneSpec(12, [-4, -4])] *2 + [LaneSpec(12, [-3, -3])] *2 ,
                   'width' :50,
                   'agent_speed_range' : (-3,-1),
                   'random_seed' : 11}]
    test_index = 0
    case = test_config[test_index]
    return gym.make('GridDriving-v0', **case)
