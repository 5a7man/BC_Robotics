## Importing necassary libraries
import gym
import numpy as np

# creating enviornment
env = gym.make('CartPole-v1')

# total number of episodes
episodes = 10

# array to store total rewards for each episode
episodes_reward = np.zeros([episodes],dtype = float)

# displaying action, observation and rewards
for episode in range(episodes):
    print("Episode "+ str(episode+1) +  " begins")
    observation = env.reset()
    for t in range(5):
        env.render()
        action = env.action_space.sample()
        print("Action for " + str(t+1) + " timestep: " + str(action))
        print("Observations for " + str(t+1) + " timestep: " + str(observation))
        observation, reward, done, info = env.step(action)
        print("Reward for " + str(t+1) + " timestep: " + str(reward))
        print('-')
        episodes_reward[episode] += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
    print('--------------')
    print("Total Reward: " + str(episodes_reward[episode]))
    print('--------------\n\n')


# closing enviornment
env.close()