# Importing necassary libraries
import gym
import numpy as np

class NeuralNetwork:
    ''' 
    A neural network controller for an AI-Gym enviornment. 
    '''
    def __init__(self,env,variance,hidden_neurons,render):
        '''
        init function to initialize variables
        '''
        self.render = render
        self.env = env
        self.pvaraince = variance
        self.nhiddens = hidden_neurons
        self.ninputs = env.observation_space.shape[0]
        if (isinstance(env.action_space, gym.spaces.box.Box)):
            self.noutputs = env.action_space.shape[0]
        else:
            self.noutputs = env.action_space.n

    def set_genotype(self,population):    
        # setting weights and baises
        temp = population[:self.nhiddens*self.ninputs]
        self.W1 = temp.reshape([self.nhiddens,self.ninputs])*self.pvaraince
        temp = population[self.nhiddens*self.ninputs:self.nhiddens*self.ninputs+self.nhiddens]
        self.b1 = temp.reshape([self.nhiddens,1])
        temp = population[self.nhiddens*self.ninputs+self.nhiddens:self.nhiddens*self.ninputs+self.nhiddens+self.nhiddens*self.noutputs]
        self.W2 = temp.reshape([self.noutputs,self.nhiddens])*self.pvaraince
        temp = population[self.nhiddens*self.ninputs+self.nhiddens+self.nhiddens*self.noutputs:]
        self.b2 = temp.reshape([self.noutputs,1])


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
                if self.render == True:
                    self.env.render()
                action = self.update(observation)
                observation, reward, done, info = self.env.step(action)
                episodes_reward[episode] += reward
                if done:
                    # print("Episode finished after {} timesteps".format(t+1))
                    break
        fitness = np.round(np.sum(episodes_reward,axis=0)/episodes,2)
        return fitness


if __name__ == '__main__':
    # connfigurations
    popsize = 10            # population size
    varaince = 0.1          # initial varaince
    ppvaraince = 0.02       # perputation varaince
    ngenerations = 100      # no of generations
    nepisodes = 3           # no of episodes

    env = gym.make('CartPole-v1')
    robot = NeuralNetwork(env = env,variance = varaince,hidden_neurons = 5,render=False)
    nparameters = robot.ninputs*robot.nhiddens+robot.noutputs*robot.nhiddens+robot.nhiddens+robot.noutputs
    population = np.random.rand(popsize,nparameters)

    # for 100 generations
    for g in range(ngenerations):
        print("--------------Generation: "+str(g+1)+"-------------")
        fitness  =[]
        for i in range(popsize):
            robot.set_genotype(population[i])
            fitness.append(robot.evaluate(episodes=nepisodes))
        best = np.argsort(np.abs(fitness))
        for i in range(int(popsize/2)):
            population[best[int(i+popsize/2)]] = population[best[i]]+np.random.rand(nparameters)*ppvaraince

    # best robot
    best_robot = NeuralNetwork(env = env,variance = varaince,hidden_neurons = 5,render=True)
    best_robot.set_genotype(population[0])
    best_robot.evaluate(episodes=50)
        