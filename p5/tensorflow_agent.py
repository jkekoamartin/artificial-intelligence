import numpy as np
import operator
from random import shuffle
import tensorflow as tf

class ContinuousTensorflowQLearningAgent(object):
    def __init__(self, observation_space_dim, action_space, hidden_dim=4,
                 learning_rate=0.1, discount=0.95, exploration_rate=0.5, exploration_decay_rate=0.99):
        tf.reset_default_graph()
        self._sess = tf.Session()
        self._discount = tf.constant(discount)
        self._state = tf.placeholder(dtype=tf.float32, shape=[1, observation_space_dim], name='observation')
        self._action = tf.placeholder(dtype=tf.int32, shape=1, name='action')
        self._next_state = tf.placeholder(dtype=tf.float32, shape=[1, observation_space_dim], name='next_observation')
        self._reward = tf.placeholder(dtype=tf.float32, shape=[1], name='reward')

        self._w1 = tf.get_variable('w1', shape=[observation_space_dim, hidden_dim],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self._b1 = tf.get_variable('b1', shape=[hidden_dim])

        self._w2 = tf.get_variable('w2', shape=[hidden_dim, action_space.n],
                                   initializer=tf.contrib.layers.xavier_initializer())
        self._b2 = tf.get_variable('b2', shape=[action_space.n])

        self._next_value = tf.reduce_max(self._compute_q(self._next_state), reduction_indices=1)
        self._q_func = self._compute_q(self._state)
        self._policy = tf.argmax(self._q_func, dimension=1)

        prev_q = tf.slice(self._q_func[0, :], self._action, [1])
        self._loss = tf.reduce_mean(tf.square(self._reward + self._discount * self._next_value - prev_q))
        self._update = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss)

        self._sess.run([tf.initialize_all_variables()])

        self._saver = tf.train.Saver()

        self._n_actions = action_space.n
        self._exploration_rate = exploration_rate
        self._exploration_decay = exploration_decay_rate


    def _compute_q(self, state):
        return tf.matmul(tf.nn.relu(tf.matmul(state, self._w1) + self._b1), self._w2) + self._b2


    def saveModel(self, path="model/model.tf"):
        self._saver.save(self._sess, path)
        print("Model saved in path: %s" % path)

    def loadModel(self, path="model/model.tf"):
        self._saver.restore(self._sess, path)
        print("Model restored from path: %s" %path)

    def reset(self):
        self._exploration_rate *= self._exploration_decay


    def act(self, observation):
        if np.random.random_sample() < self._exploration_rate:
            return np.random.randint(0, self._n_actions)
        else:
            res = self._sess.run([self._policy], feed_dict={self._state: np.array([observation])})
            return res[0][0]

    def update(self, observation, action, new_observation, reward):
        self._sess.run([self._update], feed_dict={
            self._state: np.array([observation]),
            self._action: np.array([action]),
            self._next_state: np.array([new_observation]),
            self._reward: np.array([reward])
        })

if __name__ == "__main__":
    pass