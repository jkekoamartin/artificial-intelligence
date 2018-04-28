import neurolab as nl
import numpy as np
from random import shuffle
import tensorflow as tf

class DeepRLAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.99,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99,
                 batch_size=10):
        # Create train samples
        self.input_size=observation_space_dim
        self.output_size=action_space.n
        self._batch_size=batch_size

        #define and initialize your network here
        #UNCOMMENT THESE LINES TO TEST TENSORFLOW
        #self._sess = tf.Session()
        #self._discount = tf.constant(discount)
        #self._sess.run([tf.initialize_all_variables()])

    def save(self, filename):
        raise NotImplementedError('***Error: save to file  not implemented')
        #YOUR CODE HERE: save trained model to file


    def load(self, filename):
        raise NotImplementedError('***Error: load from file not implemented')
        #YOUR CODE HERE: load trained model from file

    def reset(self):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: load trained model from file

    def act(self, observation):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action

    def update(self, observation, action, new_observation, reward):
        raise NotImplementedError('***Error: load from file not implemented')
        # YOUR CODE HERE: pick actual best action
        # Note: you may need to change the function signature as needed by your training algorithm
