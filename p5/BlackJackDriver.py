import gym
env = gym.make('Blackjack-v0')
#See details about the game rules here, but not necessary for your agent -- it will learn the rules by experimentation!
#Environment definition: https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
#actions, observations described in detail above
#so your policy network needs to learn to predict one of these actions based on the observation.

for i_episode in range(5):
    total_rewards = 0
    observation = env.reset()
    while True:
        t=0
        #env.render()  #comment out for faster training!
        print(observation)
        action = env.action_space.sample() #random action, use your own action policy here
        observation, reward, done, info = env.step(action)
        total_rewards+=reward
        t+=1
        if done:
            print("Episode finished after {} timesteps %d with reward %d ", t, total_rewards)
            break
env.close()
