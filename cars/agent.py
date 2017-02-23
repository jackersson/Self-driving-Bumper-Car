import random
from abc import ABCMeta, abstractmethod
from collections import deque

import numpy as np

from cars.utils import Action
from learning_algorithms.network import Network

import matplotlib.pyplot as plt
from matplotlib import animation

import csv

class Agent(metaclass=ABCMeta):
    @property
    @abstractmethod
    def rays(self):
        pass

    @abstractmethod
    def choose_action(self, sensor_info):
        pass

    @abstractmethod
    def receive_feedback(self, reward):
        pass


class SimpleCarAgent(Agent):
    

    def __init__(self, history_data=int(50000)):
        """
        Creates car
        :param history_data: stores previous actions 
        """
        self.evaluate_mode = False  # (True) if we evaluate model, otherwise - training mode (False)
        self._rays = 7 # ladar beams count
        # here +2 is for 2 inputs from elements of Action that we are trying to predict
        self.neural_net = Network([ self.rays + 4,    
                                    self.rays + 4,  
                                    self.rays + 4,  
                                    #self.rays + 4,        
                                    # hidden layers, example:  ((self.rays + 4) * 2)                       
                                   1],
                                   #cost function
                                   output_function=lambda x: x, output_derivative=lambda x: 1)
        self.sensor_data_history = deque([], maxlen=history_data)
        self.chosen_actions_history = deque([], maxlen=history_data)
        self.reward_history = deque([], maxlen=history_data)
        self.step = 0
        self.learning_rate = 0.04
        self.epoch_size = 30

        #shows mean reward to track training results
        #TODO make it better
        plt.ion()       
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        #self.ax = self.fig.add_axes([0, 0, 1, 1], frameon=False)
        self.line1, = self.ax.plot([], [], 'b-')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-8, 8)
        self.rewards = []        
      

    @classmethod
    def from_weights(cls, layers, weights, biases):
        """
        Creates agent with neural network params.        
        """
        agent = SimpleCarAgent()
        agent._rays = weights[0].shape[1] - 4
        nn = Network(layers, output_function=lambda x: x, output_derivative=lambda x: 1)

        if len(weights) != len(nn.weights):
            raise AssertionError("You provided %d weight matrices instead of %d" % (len(weights), len(nn.weights)))
        for i, (w, right_w) in enumerate(zip(weights, nn.weights)):
            if w.shape != right_w.shape:
                raise AssertionError("weights[%d].shape = %s instead of %s" % (i, w.shape, right_w.shape))
        nn.weights = weights

        if len(biases) != len(nn.biases):
            raise AssertionError("You provided %d bias vectors instead of %d" % (len(weights), len(nn.weights)))
        for i, (b, right_b) in enumerate(zip(biases, nn.biases)):
            if b.shape != right_b.shape:
                raise AssertionError("biases[%d].shape = %s instead of %s" % (i, b.shape, right_b.shape))
        nn.biases = biases

        agent.neural_net = nn

        return agent

    @classmethod
    def from_string(cls, s):
        from numpy import array  # special for normal eval execution
        layers, weights, biases = eval(s.replace("\n", ""), locals())
        return cls.from_weights(layers, weights, biases)

    @classmethod
    def from_file(cls, filename):
        c = open(filename, "r").read()
        return cls.from_string(c)

    def show_weights(self):
        params = self.neural_net.sizes, self.neural_net.weights, self.neural_net.biases
        np.set_printoptions(threshold=np.nan)
        return repr(params)

    def to_file(self, filename):
        c = self.show_weights()
        f = open(filename, "w")
        f.write(c)
        f.close()

    @property
    def rays(self):
        return self._rays

    def choose_action(self, sensor_info):       
        # try to predict reward for all actions that are avaliable from current state
        rewards_to_controls_map = {}

        # make discrete a values set in order to predict just some of them
        for steering in np.linspace(-1, 1, 3):  # setting discrete frequency
            for acceleration in np.linspace(-0.75, 0.75, 3): 
                action = Action(steering, acceleration)
                agent_vector_representation = np.append(sensor_info, action)
                agent_vector_representation = agent_vector_representation.flatten()[:, np.newaxis]
                predicted_reward = float(self.neural_net.feedforward(agent_vector_representation))
                rewards_to_controls_map[predicted_reward] = action

        # search for action with best reward
        rewards = list(rewards_to_controls_map.keys())
        highest_reward = max(rewards)
        best_action = rewards_to_controls_map[highest_reward]

        # Sometimes we make random action to evaluate the trained model behaviour       
        if (not self.evaluate_mode) and (random.random() < 0.05):
            highest_reward = rewards[np.random.choice(len(rewards))]
            best_action = rewards_to_controls_map[highest_reward]
            # prints the result (prediction) from network
            #print("Chosen random action w/reward: {}".format(highest_reward))
        #else:
            #print("Chosen action w/reward: {}".format(highest_reward))
     
        # store data for training step
        self.sensor_data_history.append(sensor_info)
        self.chosen_actions_history.append(best_action)
        self.reward_history.append(0.0)  #here we do not know what reward is
        # method receive_feedback calculates real reward from predicted action

        return best_action

    def receive_feedback(self, reward, train_every=50, reward_depth=7):
        """
        Receive feedback on the latest neural network decision and analyze it      
        :param reward: real world reward
        :param train_every: sufficient data count for training mode        
        :param reward_depth: how many actions in a row affects on reward        
        """
        # training intervals       
        self.step += 1

               # make reward influence on previous actions
        # so if we hit a wall we should take into account penalty in previous reward_depth steps
        i = -1
        while len(self.reward_history) > abs(i) and abs(i) < reward_depth:
            self.reward_history[i] += reward
            reward *= 0.5
            i -= 1

        #we can train network if we have sufficient amount of data
        #len(self.reward_history) >= train_every       
        if not self.evaluate_mode and (len(self.reward_history) >= train_every) and not (self.step % train_every):
                      
            X_train = np.concatenate([self.sensor_data_history, self.chosen_actions_history], axis=1)
            y_train = self.reward_history
            train_data = [(x[:, np.newaxis], y) for x, y in zip(X_train, y_train)]

            #collecting data for offline training and model evaluation
            
            with open('training_data.csv', 'a') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                for row in train_data:            
                    writer.writerow(np.append(row[0],row[1]))

            mn = np.mean(np.array(self.reward_history));    

            self.rewards.append(mn)
            items = np.arange(len(self.rewards))

            self.line1.set_data(np.arange(len(self.rewards)),self.rewards)          
            self.fig.canvas.draw()         
           
            print ("mean reward = ", mn)
            self.neural_net.SGD( training_data=train_data, epochs=self.epoch_size, 
                                 mini_batch_size=train_every , eta=self.learning_rate)

