import random
import tensorflow as tf


# Code by James Martin and Hannah Johnson

FINAL_EPSILON = 0.1     # ending/desired epsilon for training
INITIAL_EPSILON = 1.0   # starting epsilon for training
EXPLORE = 10000         # amount of steps before everything is updated
OBSERVE = 500           # observes 500 sessions before training begins
GAMMA = 0.95            # decay rate of past observations
FRAME_PER_ACTION = 1
UPDATE_TIME = 10000

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
        self.actions = tf.placeholder("float", [None])

        # Tensorflow initializations
        self._saver = tf.train.Saver()

        # define and initialize your network here
        self.createQNetwork()

        # UNCOMMENT THESE LINES TO TEST TENSORFLOW

        # saving and loading networks
        self._sess = tf.Session()
        self._discount = tf.constant(discount)
        self._sess.run([tf.initialize_all_variables()])

        checkpoint = tf.train.get_checkpoint_state("model")
        if checkpoint and checkpoint.model_checkpoint_path:
            self.load()
        else:
            print("Could not find old network weights")

        self.createTrainingMethod()

        # save every 1000 iterations
        if self.time % 1000 == 0:
            self.save("model")
        if self.time % UPDATE_TIME == 0:
            self._sess.run(self.copyTargetQNetworkOperation)


    # saves data set after 1000 training sessions
    def save(self, filename):
        self._saver.save(self._sess, 'models/' + 'network' + '-dqn', global_step=self.time)
        if self.time % UPDATE_TIME == 0:
            self._sess.run(self.copyTargetQNetworkOperation)

    # loads old data/network
    def load(self, filename):
        checkpoint = tf.train.get_checkpoint_state(filename)
        if checkpoint and checkpoint.model_checkpoint_path:
            self._saver.restore(self._sess, checkpoint.model_checkpoint_path)
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


    # chooses best action and returns it
    def act(self, observation):
        QVal = self.QValueT.eval(feed_dict={self.stateInputT: [self.currentState]})[0]
        action = tf.zeros(self.actions)
        index = 0

        if self.time % FRAME_PER_ACTION == 0:
            # if in observe state, choose random actions
            if random.random() <= self.epsilon:
                index = random.randrange(self.actions)
                action[index] = 1
            else:
                index = tf.arg_max(QVal)
            action[index] = 1
        else:
            action[0] = 1       # do nothing

        # update epsilon after observation state
        if self.epsilon > FINAL_EPSILON and self.time > OBSERVE:
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        return action

    #   # should this function be getAction or trainNetwork instead??
    # def update(self, observation, action, new_observation, reward):
    #   # YOUR CODE HERE: pick actual best action
    #   # Note: you may need to change the function signature as needed by your training algorithm


    def createQNetwork(self):

        # network weights
        conv1_weight = self.weight([8,8,4,32])
        conv1_b = self.bias([32])
        conv2_weight = self.weight([4,4,32,64])
        conv2_b = self.bias([64])
        conv3_weight = self.weight([3,3,64,64])
        conv3_b = self.bias([64])

        fc1_weight = self.weight([3136,512])
        fc1_b = self.bias([512])
        fc2_weight = self.weight([512, self.actions])
        fc2_b = self.bias([self.actions])

        # input layer
        inputState = tf.placeholder("float", [None,84,84,4])

        # hidden layers
        hidden_conv1 = tf.nn.relu(self.conv2d(inputState,conv1_weight,4) + conv1_b)
        hidden_conv2 = tf.nn.relu(self.conv2d(hidden_conv1,conv2_weight,2) + conv2_b)
        hidden_conv3 = tf.nn.relu(self.conv2d(hidden_conv2,conv3_weight,1) + conv3_b)
        hidden_conv3_shape = hidden_conv3.get_shape().as_list()
        hidden_conv3_flat = tf.reshape(hidden_conv3, [-1,3136])
        hidden_fc1 = tf.nn.relu(tf.mathmul(hidden_conv3_flat,fc1_weight) + fc1_b)

        # Q Value output layer
        QVal = tf.matmul(hidden_fc1,fc2_weight) + fc2_b

        return (inputState, QVal, conv1_weight, conv1_b, conv2_weight, conv2_b, conv3_weight, conv3_b, fc1_weight, fc1_b, fc2_weight, fc2_b)

    def createTrainingMethod(self):
        self.actionInput = tf.placeholder("float", [None, self.actions])
        self.yInput = tf.placeholder("float",[None])

        # Q learning equation, cost function, training optimizer
        QAction = tf.reduce_sum(self.QVal*self.actionInput, 1)
        self.cost = tf.reduce_mean(tf.square(self.yInput - QAction))
        self.trainStep = tf.trainStep = tf.train.RMSPropOptimizer(0.0025,0.99,0.0,1e-6).minimize(self.cost)
