import sys
import os
import time

import traci
sys.path.append(os.getcwd())
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym
import gym_examples
from gym_examples.envs.sumo_env import SumoEnv
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter
import torch


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


writer = SummaryWriter('log_dir_0903')


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of all previous scores')
    #plt.title('Running average of previous 25 scores')
    plt.savefig(figure_file)

########### Hipper parameters ###########
n_observations = 2
n_actions = 1
lr = 5e-4
n_epochs = 8 #5
save_dir = "log_dir_0903"
batch_size = 256 # 64*10
n_episodes = 15
episode = 1
##########################################

agent = Agent(n_observations=n_observations, n_actions=n_actions, lr=lr,
               save_dir=save_dir, n_epochs=n_epochs, batch_size=batch_size)

total_step = 0
# agent.load_models()


for k in range (1, 500): # run 500 learning iterations
    for j in range(1,301): # moving through the sumo config files from 1 to 300
        score_history = []
        sumoConfig= f"sumo_files/sumoconfig{j}.sumocfg"
        # check if the sumo config file exists
        if not os.path.exists(sumoConfig):
            print(f"sumo config file {sumoConfig} does not exist")
        env = gym.make("gym_examples/Sumo", sumoConfig=sumoConfig)
    #for i in range(1, n_episodes): # run each sumo config file for 15 episodes
        observation, info = env.reset()
        #env.sumo_simulation = f"sumo_files/sumoconfig{j}.sumocfg"
        done = False
        score = 0
        step = 0
        start_episode = time.time()
        while not done:
            #print("time:", traci.simulation.getTime())
            action, log_probs, val = agent.choose_action(observation)
            observation_, reward, terminated, trancuated, info = env.step(action) 
            step += 1
            total_step += 1
            score +=  reward
            done = terminated or trancuated
            if done:
                print("episode done, step: ", step)
            agent.remember(observation, action, log_probs, val, reward, done)
            observation = observation_
        
        env.close()
        print("step: ", step)
        end_episode = time.time()
        score_history.append(score)
        writer.add_scalar('Performance/Reward', score, episode)
        print(f"episode: {episode}, score: {score:.2f}")
        print(f"episode time: {end_episode - start_episode:.2f}")
        episode += 1
        if (j+1) % 20 == 0: # learn each 5 episodes
            start_learning = time.time() 
            agent.learn()
            print("time of learning: ", time.time() - start_learning)
            agent.save_models()
        

        avg_score = np.array(score_history).mean()
        print(f"average score: {avg_score:.0f}")

# plot the learning curve after all the learning process
# x = [i+1 for i in range(len(score_history))]
# figure_file = 'leraning_curve.png'
# plot_learning_curve(x, score_history, figure_file)
# env.close()
# writer.close()









