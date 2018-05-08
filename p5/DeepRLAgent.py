import random
import tensorflow as tf


# Code by James Martin and Hannah Johnson

FINAL_EPSILON = 0.1     # ending/desired epsilon for training
INITIAL_EPSILON = 1.0   # starting epsilon for training
EXPLORE = 10000         # amount of steps before everything is updated
OBSERVE = 500           # observes 500 sessions before training begins
GAMMA = 0.95            # decay rate of past observations
FRAME_PER_ACTION = 1


# Tensorflow stuff is used by this agent
# We might consider using Tensoflow slim -- handles the nitty gritty of defining the network
class DeepRLAgent(object):
    def __init__(self, observation_space_dim, action_space,
                 learning_rate=0.1,
                 discount=0.99,
                 exploration_rate=0.5,
                 exploration_decay_rate=0.99,
                 batch_size=10):
        # Create train samples
        self.input_size = observation_space_dim
        # note: using .n means we can only pass a Discrete type action space
        self.output_size = action_space.n
        self._batch_size = batch_size
        self.time = 0

        # Tensorflow initializations
        self._saver = tf.train.Saver()
        # define and initialize your network here
        # UNCOMMENT THESE LINES TO TEST TENSORFLOW
        self._sess = tf.Session()
        self._discount = tf.constant(discount)
        self._sess.run([tf.initialize_all_variables()])


    # saves data set after 1000 training sessions
    def save(self, filename):
        self.saver = tf.train.Saver()
        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())


    # loads old data/network
    def load(self, filename):
        checkpoint = tf.train.get_checkpoint_state("models")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.session, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
        else:
            print("Could not find old network weights")

    def reset(self):
        self.load("model")
        self.epsilon = INITIAL_EPSILON

        # TODO: still directly from refRL, i needa figure out what this is about
        self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T = self.createQNetwork()

        self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1), self.b_conv1T.assign(self.b_conv1),
                                            self.W_conv2T.assign(self.W_conv2), self.b_conv2T.assign(self.b_conv2),
                                            self.W_conv3T.assign(self.W_conv3), self.b_conv3T.assign(self.b_conv3),
                                            self.W_fc1T.assign(self.W_fc1), self.b_fc1T.assign(self.b_fc1),
                                            self.W_fc2T.assign(self.W_fc2), self.b_fc2T.assign(self.b_fc2)]

        self.createTrainingMethod()

    # chooses best action and returns it
    def act(self, observation):
        QVal = self.QValueT.eval(feed_dict={self.stateInputT: [self.currentState]})[0]
        action = tf.zeros(self.actions)
        index = 0
        if self.time % FRAME_PER_ACTION == 0:
            if random.random() <= self.epsilon:
                index = random.randrange(self.actions)
                action[index] = 1
            else:
                index = tf.argmax(QVal)
                action[index] = 1
        else:
            action[0] = 1  # don't do anything

        # update epsilon after 10000 steps
        if self.epsilon > FINAL_EPSILON and self.time > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action

    # todo: need to implement
    def update(self, observation, action, new_observation, reward):
        raise NotImplementedError('***Error: load from file not implemented')

        # YOUR CODE HERE: pick actual best action
        # Note: you may need to change the function signature as needed by your training algorithm
