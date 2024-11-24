import sys
import os
sys.path.append(os.getcwd())
import numpy as np
import gymnasium as gym
import gym_examples
from gym_examples.envs.sumo_env import SumoEnv
from ppo import Agent
from torch.utils.tensorboard import SummaryWriter
import torch
import time
#from torch.profiler import profile, record_function, ProfilerActivity

writer = SummaryWriter('log_dir')
env = gym.make("gym_examples/Sumo")

n_observations = 28
n_actions = 24
lr = 5e-4
n_epochs = 5
save_dir = "save_dir"
batch_size = 64
n_episodes = 1
episode = 1
agent = Agent(n_observations=n_observations, n_actions=n_actions, lr=lr,
               save_dir=save_dir, n_epochs=n_epochs, batch_size=batch_size)

total_step = 0
# agent.load_models()

reset_times = []
episode_times = []
start_training_time = time.time()
for _ in range(100):
    #print('hey')
    score_history = []
    for _ in range(n_episodes):
        env_reset_start_time = time.time()  # Start timing the reset

        observation, info = env.reset()
        
        env_reset_end_time = time.time()  # End timing the reset
        env_reset_duration = env_reset_end_time - env_reset_start_time
        reset_times.append(env_reset_duration)  # Append time to the list
        done = False
        
        score = 0
        step = 0

        # Timing for the entire episode
        episode_start_time = time.time()
        
        while not done:

            action, log_probs, val = agent.choose_action(observation)
            observation_, reward, terminated, trancuated, info = env.step(action) 
            step += 1
            total_step += 1
            score +=  reward
            done = terminated or trancuated
            agent.remember(observation, action, log_probs, val, reward, done)
            observation = observation_
        
        episode_end_time = time.time()  # End timing for the episode
        episode_duration = episode_end_time - episode_start_time
        episode_times.append(episode_duration)
        
        print(f"Total time for episode: {episode_duration:.2f} seconds")

        # Optionally, print or log the iteration times for the episode
        #print(f"Iteration times for episode {episode}: {iteration_times}")
        
        print('end while')
        score_history.append(score)
        writer.add_scalar('Performance/Reward', score, episode)
        print(f"episode: {episode}, score: {score:.2f}")
        episode += 1

    avg_score = np.array(score_history).mean()
    print(f"average score: {avg_score:.0f}")

    
    agent.learn()
    agent.save_models()

end_training_time = time.time()
training_time_duration = end_training_time - start_training_time
print(f"Total time for simulation: {training_time_duration:.2f} seconds")

env.close()
writer.close()









