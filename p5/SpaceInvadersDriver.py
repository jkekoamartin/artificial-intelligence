import matplotlib.pyplot as plt

import cv2
import sys
from p5 import DeepRLAgent
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

    agent = DeepRLAgent.DeepRLAgent(actions)

    action = 0  # do nothing
    observation0, reward, terminal, info = env.step(action)
    print("Before processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(observation0))
    plt.show()
    observation0 = preprocess(observation0)
    print("After processing: " + str(np.array(observation0).shape))
    plt.imshow(np.array(np.squeeze(observation0)))
    plt.show()

    agent.setInitState(observation0)
    agent.currentState = np.squeeze(agent.currentState)

    while True:
        action = agent.getAction()
        max_action = np.argmax(np.array(action))
        if terminal:
            env.reset()

        next_observation, reward, terminal, info = env.step(max_action)
        env.render()
        next_observation = preprocess(next_observation)
        agent.set_next_state(next_observation, action, reward, terminal)


def test():
    # this loads the model, and uses an agent to run the game



    env = gym.make('SpaceInvaders-v0')
    env.reset()
    actions = env.action_space.n

    agent = DeepRLAgent.DeepRLAgent(actions)

    action = 0  # do nothing
    observation, reward, terminal, info = env.step(action)

    observation = preprocess(observation)

    agent.setInitState(observation)
    agent.currentState = np.squeeze(agent.currentState)

    plays = 0
    test_flag = True

    while plays < 10:
        action = agent.getAction()
        max_action = np.argmax(np.array(action))
        if terminal:
            env.reset()
            plays += 1

        next_observation, reward, terminal, info = env.step(max_action)
        env.render()
        next_observation = preprocess(next_observation)
        agent.set_next_state(next_observation, action, reward, terminal, test_flag)


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


if __name__ == "__main__":
    # check correct length args

    # for ease of development, this runs train when there are no args

    if len(sys.argv) is 1:
        # if no argument, just train
        test()
    else:
        error = "Invalid arguments passed. Please input \"-train\" or \"-test\" "
        if len(sys.argv) is 2:
            if sys.argv[1] == "-test":
                print("test")
                test()
            elif sys.argv[1] == "-train":
                train()
            else:
                print(error)
        else:
            print(error)
