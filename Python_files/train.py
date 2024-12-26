import sys
import os
sys.path.append(os.getcwd())
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
import gym_examples
from gym_examples.envs.sumo_env import SumoEnv
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter
import torch
#from torch.profiler import profile, record_function, ProfilerActivity


class Logger:
    def __init__(self, file_path):
        self.terminal = sys.stdout  # Keep the original stdout
        self.log_file = open(file_path, "a")  # Open a file for appending

    def write(self, message):
        self.terminal.write(message)  # Print to the console
        self.log_file.write(message)  # Write to the file

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

# Redirect stdout to Logger
sys.stdout = Logger("log2312")

writer = SummaryWriter('log_dir')
env = gym.make("gym_examples/Sumo")

def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of all previous scores')
    #plt.title('Running average of previous 25 scores')
    plt.savefig(figure_file)

########### Hipper parameters ###########
n_observations = 8 
n_actions = 1
lr = 3e-4 #5e-4
n_epochs = 5
#save_dir = "save_dir"
save_dir = "log_dir_2312"
batch_size = 64
n_episodes = 25
episode = 1
##########################################

agent = Agent(n_observations=n_observations, n_actions=n_actions, lr=lr,
               save_dir=save_dir, n_epochs=n_epochs, batch_size=batch_size)

total_step = 0
# agent.load_models()

for _ in range(500): 
    score_history = []
    # run from 1 to 25 episodes
    for i in range(1, n_episodes):
        observation, info = env.reset()
        done = False
        score = 0
        step = 0
        
        while not done:
            action, log_probs, val = agent.choose_action(observation)
            observation_, reward, terminated, trancuated, info = env.step(action) 
            step += 1
            total_step += 1
            score +=  reward
            done = terminated or trancuated
            agent.remember(observation, action, log_probs, val, reward, done)
            observation = observation_
        
        score_history.append(score)
        writer.add_scalar('Performance/Reward', score, episode)
        print(f"episode: {episode}, score: {score:.2f}")
        episode += 1
        if i % 1 == 0: # learn each 3 episodes
            agent.learn()
            agent.save_models()

    avg_score = np.array(score_history).mean()
    print(f"average score: {avg_score:.0f}")

    
    #agent.learn() # learning after n_episodes
    #agent.save_models()

# plot the learning curve after all the learning process
x = [i+1 for i in range(len(score_history))]
figure_file = 'leraning_curve.png'
plot_learning_curve(x, score_history, figure_file)
env.close()
writer.close()









