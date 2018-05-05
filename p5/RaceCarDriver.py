import matplotlib.pyplot as plt

import cv2
import sys
from p5 import refRL
import numpy as np

import gym



#################################################
# public methods
#################################################
def train():
    # this processes frame, runs the learning agent, and saves the model
    # print("train")

    env = gym.make('SpaceInvaders-v0')
    env.reset()
    actions = env.action_space.n
    print(actions)

    brain = refRL.BrainDQN(actions)

    action0 = 0  # do nothing
    observation0, reward0, terminal, info = env.step(action0)
    print("Before processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(observation0))
    plt.show()
    observation0 = preprocess(observation0)
    print("After processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(np.squeeze(observation0)))
    plt.show()

    brain.setInitState(observation0)
    brain.currentState = np.squeeze(brain.currentState)

    while True:
        action = brain.getAction()
        actionmax = np.argmax(np.array(action))
        if terminal:
            nextObservation = env.reset()

        nextObservation, reward, terminal, info = env.step(actionmax)
        env.render()
        nextObservation = preprocess(nextObservation)
        brain.setPerception(nextObservation, action, reward, terminal)
    # for _ in range(1000):
    #     env.render()
    #     action = env.action_space.sample()  # your agent here (this takes random actions)
    #     print(action)
    #     observation, reward, done, info = env.step([0, 0, 0])
    #     for each in observation:
    #         print(each)

    # testing processing
    ##########
    # action0 = 0  # do nothing
    # observation0, reward0, terminal, info = env.step(action0)
    # print("Before processing: " + str(np.array(observation0).shape))
    # plt.imshow(np.array(observation0))
    # plt.show()
    # observation0 = _p_process(observation0)
    # print("After processing: " + str(np.array(observation0).shape))
    # plt.imshow(np.array(np.squeeze(observation0)))
    # plt.show()
    ##########






def test():
    # this loads the model, and uses an agent to run the game

    pass


#################################################
# private methods
#################################################
def preprocess(observation):
    """
    Pre-processes observation for use by neural network
    :param observation:
    raw image
    :return:
    processed image
    """
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_BGR2GRAY)
    observation = observation[26:110, :]
    ret, observation = cv2.threshold(observation, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(observation, (84, 84, 1))

# env = gym.make('CarRacing-v0')
# #Details about environment: https://github.com/openai/gym/blob/master/gym/envs/box2d/car_racing.py
# env.reset()
# total_reward = 1000
# for i_episode in range(2):
#     observation = env.reset()
#     for t in range(1000000):
#         env.render()
#
#         #action = env.action_space.sample()
#         #self.action_space = spaces.Box( np.array([-1,0,0]), np.array([+1,+1,+1]))  # steer, gas, brake
#         #self.observation_space = spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8)
#         #print(action)
#         action=[0,1,0]  #forward, gas, no break
#         observation, reward, done, info = env.step(action)
#         print (reward, done, info)
#         total_reward += reward
#
#         #NOTE: Need to detect if off-the-road!!! car will continue to "drive" on grass
#         if done:
#             print("Episode finished after {} timesteps".format(t+1))
#             env.reset()
#             total_reward += -1000
#             break
# env.close()

if __name__ == "__main__":
    # check correct length args

    # for ease of development, this runs train when there are no args

    if len(sys.argv) is 1:
        train()
    else:
        error = "Invalid arguments passed. Please input \"-train\" or \"-test\" "
        if len(sys.argv) is 2:
            if sys.argv[1] == "-test":
                test()
            elif sys.argv[1] == "-train":
                train()
            else:
                print(error)
        else:
            print(error)
