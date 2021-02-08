import gym
import numpy as np
from gym.envs.registration import register

env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'FrozenLake' in env:
        del gym.envs.registration.registry.env_specs[env]
        
register(
  id='FrozenLake-v3',
  entry_point = 'gym.envs.toy_text:FrozenLakeEnv',
  kwargs={'map_name':'4x4',
          'is_slippery':False}
)

env = gym.make('FrozenLake-v3')
# initialize Q-Table
Q = np.zeros([env.observation_space.n,env.action_space.n]) #16*4 matrix

# set learning parameter
lr = .1 #learning rate(학습률)
y = 0.99 #discount rate(감가율)
num_episodes = 2000

# create lists to contain total rewards per episode
rList = [] #에피소드별 총 리워드 저장하는 리스트

for i in range(num_episodes):
    # Reset environment and get first new observation
    state = env.reset()
    rAll = 0
    done = False

    # The Q-Table learning algorithm
    while not done:
        # Choose an action by greedily (with noise) picking from Q table
        action = np.argmax(Q[state,:] + np.random.randn(1,env.action_space.n) / (i+1))

        # Get new state and reward from environment
        new_state, reward, done,_ = env.step(action)

        # Get negative reward every step
        if reward == 0 :
            reward=-0.001

        # Q-Learning
        Q[state,action]= Q[state,action]+lr*(reward+y* np.max(Q[new_state,:])-Q[state,action])
        state = new_state
        rAll += reward
 
    rList.append(rAll)


print("Success rate : "+str(sum(rList) / num_episodes))