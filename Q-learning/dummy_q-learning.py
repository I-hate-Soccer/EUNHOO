# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 15:09:50 2021

@author: 이은후
"""

import gym
import numpy as np
from gym.envs.registration import register
import random as pr

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FrozenLake' in env:
        del gym.envs.registration.registry.env_specs[env]

def qmax_action(four_q):
    """ 상태 s 에서 네가지 a 에 따른 네가지 Q 중 가장 큰 것을 선택 (같으면 랜덤하게 선택)"""
    maxq = np.amax(four_q)
    indices = np.nonzero(four_q == maxq)[0]
    return pr.choice(indices)


register(
    id='FrozenLake-v3',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name': '4x4',
            'is_slippery': False}
)

env = gym.make('FrozenLake-v3')

# shape = [States num, 4(left,down,right,up)]
Q = np.zeros([env.observation_space.n, env.action_space.n])
# Set learning parameters
num_episodes = 2000

# create lists to contain total rewards and steps per episode
rList = []
for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        action = qmax_action(Q[state, :])

        # Get new state and reward from environment
        new_state, reward, done, _ = env.step(action)

        # Update Q-Table with new knowledge using learning rate
        Q[state, action] = reward + np.max(Q[new_state, :])

        rAll += reward
        state = new_state

    rList.append(rAll)

print("Success rate: " + str(sum(rList) / num_episodes))
