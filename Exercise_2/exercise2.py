# Importing necassary libraries
import gym
import numpy as np

class NeuralNetwork:
    ''' 
    A neural network controller for an AI-Gym enviornment. 
    '''
    def __init__(self,env,variance,hidden_neurons):
        '''
        init function to initialize variables
        '''
        self.env = env
        self.pvaraince = variance
        self.nhiddens = hidden_neurons
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n

        # weights and baises
        self.W1 = np.random.randn(self.nhiddens,self.ninputs)*self.pvaraince
        self.b1 = np.zeros((self.nhiddens,1),dtype=float)
        self.W2 = np.random.randn(self.noutputs,self.nhiddens)*self.pvaraince
        self.b2 = np.zeros((self.noutputs,1),dtype=float)

    def update(self,observation):
        '''
        To update action
        '''
        observation.resize(self.ninputs,1)
        Z1 = np.dot(self.W1,observation)+self.b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(self.W2,A1)+self.b2
        A2 = np.tanh(Z2)

        if (isinstance(self.env.action_space, gym.spaces.box.Box)):
            action = A2
        else:
            action = np.argmax(A2)

        return action
        
    def evaluate(self,episodes):
        '''
        Rendering and calculating rewards
        '''
        episodes_reward = np.zeros([episodes],dtype = float)
        for episode in range(episodes):
            observation = self.env.reset()
            for t in range(100):
                self.env.render()
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                episodes_reward[episode] += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
            episodes_reward[episode] = np.round(episodes_reward[episode]/(t+1),2)
        print("Rewards for "+str(episodes)+" episodes "+str(episodes_reward))


if __name__ == '__main__':
    env = gym.make('Pendulum-v1')
    for _ in range(100):
        print('---------- Robot ' + str(_+1) +'----------')
        robot = NeuralNetwork(env = env,variance = 0.01,hidden_neurons = 5)
        robot.evaluate(episodes = 10)
        print('-------------------------\n\n')