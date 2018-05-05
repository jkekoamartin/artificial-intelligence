""" Trains an agent with (stochastic) Policy Gradients on various Atari games. """
# written April 2017 by Guntis Barzdins and Renars Liepins
# inspired by Karpathy's gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
import numpy as np
import gym
import tensorflow as tf
import _pickle as cPickle
import types
from IPython.display import HTML, display
import base64
import matplotlib.pyplot as plt
% matplotlib
inline

# hyperparameters
H = 50  # low neuron count forces generalisation over memorisation
batch_size = 10  # every how many episodes to do a param update?
learning_rate = 1e-3
gamma = 0.99  # discount factor for reward
decay_rate = 0.99  # decay factor for RMSProp leaky sum of grad^2
resume = False  # resume from previous checkpoint?
D = 105 * 80  # input dimensionality: 105x80 grid, now it is just half resolution
np.random.seed(2)  # make experiments reproducible

env = gym.make("PongDeterministic-v3")  # environment info


# env = gym.make("SpaceInvaders-v3") # environment info
# env = gym.make("Breakout-v3") # environment info
# env = gym.make("Boxing-v3")
# env = gym.make("Pong-v3")
# env = gym.make("Bowling-v3")
# env = gym.make("Seaquest-v3")
# env = gym.make("Robotank-v3")
# env = gym.make("MontezumaRevenge-v3")
# env = gym.make("Frostbite-v3")

# helper function for showing video inline
def show_video(fname, mimetype):
    video_encoded = base64.b64encode(open(fname, "rb").read()).decode('ascii')
    video_tag = '<video controls alt="test" src="data:video/{0};base64,{1}">'.format(mimetype, video_encoded)
    display(HTML(data=video_tag))


def _step_custom(self, a):
    env = self.unwrapped or self
    action = env._action_set[a]
    num_steps = 4  # self.np_random.randint(2, 5)
    ob = [];
    reward = 0.0
    for _ in range(num_steps):
        reward += env.ale.act(action)
        ob.append(env._get_obs())
    ob = np.maximum.reduce(ob)  # returns max pixel values from 4 frames
    return ob, reward, env.ale.game_over(), {"ale.lives": env.ale.lives()}  # lives left


def prepro(I):
    I = I[::2, ::2, 2]  # downsample by factor of 2, choose colour 2 to improve visibility in other games
    I[I == 17] = 0  # erase background (background type 1)
    I[I == 192] = 0  # erase background (background type 2)
    I[I == 136] = 0  # erase background (background type 3)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    # plt.imshow(I)
    # plt.show()
    return I.astype(np.float).ravel()  # 2D array to 1D array (vector)


def softmax(x):
    exps = np.exp(x - x.max())
    return exps / np.sum(exps)


def policy_forward(x):
    h = np.dot(model['W1'], x)
    h[h < 0] = 0  # ReLU nonlinearity
    logp = np.dot(model['W2'], h)
    p = softmax(logp)
    return p, h


def policy_backward(eph, epdlogp):
    dW2 = np.dot(eph.T, epdlogp).T
    dh = np.dot(epdlogp, model['W2'])
    dh[eph <= 0] = 0  # backpro ReLU
    dW1 = np.dot(dh.T, epx)
    return {'W1': dW1, 'W2': dW2}


def discount_rewards(r):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        if r[t] != 0: running_add = 0  # reset the sum, since this was a game boundary (pong specific!)
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


env = gym.wrappers.Monitor(env, '/tmp/atari_test', force=True, video_callable=lambda count: True)
env.unwrapped._step = types.MethodType(_step_custom, env)
valid_actions = range(env.action_space.n)  # number of valid actions in the specific game
print('action count:', env.action_space.n)

# model initialization    
if resume:
    model = cPickle.load(open('save.p', 'rb'))
    print('MODEL RESTORED FROM THE CHECKPOINT FILE')
else:
    model = {}
    model['W1'] = np.random.randn(H, D) / np.sqrt(D)  # "Xavier" initialization
    model['W2'] = np.random.randn(len(valid_actions), H) / np.sqrt(H)  ##added dimension for more actions
grad_buffer = {k: np.zeros_like(v) for k, v in model.items()}  # update gradient buffers over a batch
rmsprop_cache = {k: np.zeros_like(v) for k, v in model.items()}  # rmsprop memory

observation = env.reset()
prev_x = None  # used in computing the difference frame
lives = None
xs, hs, dlogps, drs = [], [], [], []
running_reward = None
reward_sum = 0
episode_number = 1
lgraphx, lgraphy = [], []  # used for plotting graph
while True:
    # preprocess the observation, set input to network to be difference image
    cur_x = prepro(observation)
    x = cur_x - prev_x if prev_x is not None else np.zeros(D)
    prev_x = cur_x

    # forward the policy network and sample an action from the returned probability
    aprob, h = policy_forward(x)
    action = np.random.choice(len(aprob), p=aprob)
    # record various intermediates (needed later for backprop)
    xs.append(x)  # observation
    hs.append(h)  # hidden state
    y = np.zeros(len(aprob));
    y[action] = 1  # one-hot encoding for the chosen action
    dlogps.append(y - aprob)  # grad that encourages the action that was taken to be taken

    # step the environment and get new measurements
    observation, reward, done, info = env.step(action)
    reward_sum += reward
    drs.append(reward)  # record reward (has to be done after we call step() to get reward for previous action)
    if lives == None: lives = info["ale.lives"]
    if info["ale.lives"] < lives: done = True  # Terminate the episode after first life lost

    if done:  # an episode finished
        # stack together all inputs, hidden states, action gradients, and rewards for this episode
        epx = np.vstack(xs)
        eph = np.vstack(hs)
        epdlogp = np.vstack(dlogps)
        epr = np.vstack(drs)
        xs, hs, dlogps, drs = [], [], [], []  # reset array memory

        # compute the discounted reward backwards through time
        discounted_epr = discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        tmp = np.std(discounted_epr)
        if tmp > 0: discounted_epr /= tmp  # fixed occasional zero-divide

        epdlogp *= discounted_epr  # modulate the gradient with advantage (PG magic happens right here.)
        grad = policy_backward(eph, epdlogp)
        for k in model: grad_buffer[k] += grad[k]  # accumulate grad over batch

        # boring book-keeping
        accumulate = min([episode_number, 100])
        running_reward = reward_sum if running_reward is None else running_reward * (
                    1 - 1 / accumulate) + reward_sum * (1 / accumulate)
        print('episode: {}, reward: {}, mean reward: {:3f}'.format(episode_number, reward_sum, running_reward))
        lgraphx.append(episode_number)
        lgraphy.append(running_reward)

        # perform rmsprop parameter update every batch_size episodes
        if episode_number % batch_size == 0:
            plt.plot(lgraphx, lgraphy)
            plt.show()
            for k, v in model.items():
                g = grad_buffer[k]  # gradient
                rmsprop_cache[k] = decay_rate * rmsprop_cache[k] + (1 - decay_rate) * g ** 2
                model[k] += learning_rate * g / (np.sqrt(rmsprop_cache[k]) + 1e-5)
                grad_buffer[k] = np.zeros_like(v)  # reset batch gradient buffer

        if episode_number % 100 == 0:
            cPickle.dump(model, open('save.p', 'wb'))
            print('MODEL SAVED TO THE CHECKPOINT TO FILE')
        reward_sum = 0
        observation = env.reset()  # reset env
        prev_x = None
        lives = None
        episode_number += 1