### Train the model
import os
import gym
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from ppo import Agent
from sumo_env import SumoEnv
import sys
import logging
from torch.utils.tensorboard import SummaryWriter
import torch

writer = SummaryWriter('log_dir') # create a tensorboard writer


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


if not os.path.exists('tmp'): # create a directory to save models
    os.makedirs('tmp')

env = gym.make('SumoEnv-v0') # create an environment

########### Hipper parameters ###########
N = 20 # learning frequency
batch_size = 64
n_epochs = 10
alpha = 3e-4 # learning rate
policy_clip = 0.2 # PPO clip parameter
lamda = 0.95 # GAE, Generalized Advantage Estimation
gamma = 0.99 # discount factor
capacity = 5e5 # memory capacity
##########################################
agent = Agent(state_dim=env.observation_space.shape,
              action_dim=env.action_space.n, 
              batch_size=batch_size,
              n_epochs=n_epochs,
              policy_clip=policy_clip,
              gamma=gamma,
              lamda=lamda, 
              adam_lr=alpha,
              capacity=capacity)
n_games = 60
figure_file = 'leraning_curve.png'
best_score = env.reward_range[0]
score_history = []
learn_iters = 0
avg_score = 0
n_steps = 0
for i in range(n_games):
    current_state,info = env.reset()
    terminated,truncated = False,False
    done = False
    score = 0
    while not done:
        action, prob, val = agent.choose_action(current_state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = 1 if (terminated or truncated) else 0
        n_steps += 1
        score += reward
        agent.store_data(current_state, action, prob, val, reward, done)
        #if n_steps % N == 0:
        #    agent.learn()
        #    learn_iters += 1
        current_state = next_state
        # maybe save the model here- the difference is that here it saves the model every 20 steps
    score_history.append(score)
    avg_score = np.mean(score_history[-100:]) # calculate the average score of the last 100 episodes
    #if avg_score > best_score: # save the best model
    #    best_score = avg_score 
    agent.save_models() # save the model after each episode, saving here might cause to model to not learn well
    writer.add_scalar('Performance/Reward', score, i) # log the reward to tensorboard
    print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
            'time_steps', n_steps, 'learning_steps', learn_iters)
    env.close()
    if i % 3 == 0:
        agent.learn()
        learn_iters += 1
    
    
x = [i+1 for i in range(len(score_history))]
plot_learning_curve(x, score_history, figure_file)