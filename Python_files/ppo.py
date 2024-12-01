import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli

from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


class Memory:
    """
    A class to store and manage experience replay memory.
    operates as a FIFO - when reaching full capacity the first sample is overriden.
    Attributes:
    batch_size (int): The size of each batch.
    capacity (int): The maximum number of samples to store - buffer size.

    """
    def __init__(self, batch_size, capacity):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
        self.capacity = capacity

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        if len(self.states) >= self.capacity:
            self.states.pop(0)
            self.actions.pop(0)
            self.probs.pop(0)
            self.vals.pop(0)
            self.rewards.pop(0)
            self.dones.pop(0)   
        
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
        
    def clear_memory(self):
        # optional method to clear memory, not needed when using specific buffer capacity
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class PolicyNetwork(nn.Module):
    """
    Policy neural network.

    Attributes:
    checkpoint_file (str): Path to save/load the model.
    actor: The actor network architecture.
    optimizer: The optimizer used for gradient descent.
    device (torch.device): The device for calculations (CPU/GPU).
    """
    def __init__(self, n_actions, n_observations, lr, save_dir, num_cells=256):
        super(PolicyNetwork, self).__init__()

        self.checkpoint_file = os.path.join(save_dir, 'policy_net15.pth')
        self.actor = nn.Sequential(
                nn.Linear(n_observations, num_cells),
                nn.ReLU(),
                nn.Linear(num_cells, num_cells),
                nn.ReLU(),
                nn.Linear(num_cells, n_actions),
                nn.Sigmoid()
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        probs = self.actor(observation)
        dist = Bernoulli(probs=probs)
        
        return dist

    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))


class ValueNetwork(nn.Module):
    """
    Value/Critic neural network.

    Attributes:
    checkpoint_file (str): Path to save/load the model.
    critic: The value/critic network architecture.
    optimizer: The optimizer used for gradient descent.
    device (torch.device): The device for calculations (CPU/GPU).
    """
    def __init__(self, n_observations, lr, save_dir, num_cells=256):
        super(ValueNetwork, self).__init__()

        self.checkpoint_file = os.path.join(save_dir, 'value_net15.pth')
        self.critic = nn.Sequential(
                nn.Linear(n_observations, num_cells),
                nn.ReLU(),
                nn.Linear(num_cells, num_cells),
                nn.ReLU(),
                nn.Linear(num_cells, 1)
        )

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        value = self.critic(state)
        return value

    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))


class Agent:
    """
    Agent for training and interacting with the environment.

    Attributes:
    gamma (float): Discount factor for future rewards.
    epsilon_clip (float): Clipping value for PPO.
    n_epochs (int): Number of epochs for training.
    gae_lambda (float): Lambda for GAE.
    global_epoch (int): Counter for epochs. used to log data
    global_batch (int): Counter for batches. used to log data
    n_actions (int): Number of possible actions.
    n_observations (int): Number of observations.
    actor (PolicyNetwork): The policy network.
    critic (ValueNetwork): The value network.
    memory (Memory): The experience replay memory.

    Methods:
    remember(observation, action, probs, vals, reward, done): Stores data in memory.
    save_models(): Saves the actor and critic models.
    load_models(): Loads the actor and critic models.
    choose_action(observation): Chooses an action based on the observation.
    learn(): Trains the agent using stored memory.
    """
    def __init__(self, n_actions, n_observations, save_dir, gamma=0.99, lr=3e-4, gae_lambda=0.95,
            epsilon_clip=0.2, batch_size=64, n_epochs=10, capacity=5e5):
        self.gamma = gamma
        self.epsilon_clip = epsilon_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.global_epoch = 0
        self.global_batch = 0
        self.n_actions = n_actions
        self.n_observations = n_observations

        self.actor = PolicyNetwork(n_actions, n_observations, lr, save_dir=save_dir)
        self.critic = ValueNetwork(n_observations, lr, save_dir=save_dir)
        self.memory = Memory(batch_size, capacity)
       
    def remember(self, observation, action, probs, vals, reward, done):
        #print('remmember')
        self.memory.store_memory(observation, action, probs, vals, reward, done)

    def save_models(self):
        print('saving models')
        self.actor.save_model()
        self.critic.save_model()

    def load_models(self):
        print('loading models')
        self.actor.load_model()
        self.critic.load_model()

    def choose_action(self, observation):
        #print('chooseAct')
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)
        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()
        action = torch.squeeze(action)
        log_probs = torch.squeeze(dist.log_prob(action))             
        value = torch.squeeze(value).item()

        # Convert to numpy arrays
        action_np = action.detach().cpu().numpy()
        log_probs_np = log_probs.detach().cpu().numpy()
        return action_np, log_probs_np, value

    def learn(self):
        print('learn')
        writer_1 = SummaryWriter('log_dir')

        for _ in range(self.n_epochs):
            
            state_arr, action_arr, old_log_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1):
                discount = 1
                a_t = 0
                for k in range(t, len(reward_arr)-1):
                    if dones_arr[k]:  # Check if the state is terminal
                        break  # Do not include value of next state if current state is terminal
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k])
                    discount *= self.gamma*self.gae_lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device)

            

            values = torch.tensor(values).to(self.critic.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_log_probs = torch.tensor(old_log_prob_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)
                
                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = torch.squeeze(critic_value)

                new_log_probs = dist.log_prob(actions)
                new_log_prob = torch.sum(new_log_probs, dim=1)
                old_log_prob = torch.sum(old_log_probs, dim=1)
                prob_ratio = new_log_prob.exp() / old_log_prob.exp()

                advantage_reshaped = advantage[batch]
                weighted_probs = advantage_reshaped * prob_ratio
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.epsilon_clip,
                        1+self.epsilon_clip)*advantage_reshaped
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs)
                # averaging over all batch
                actor_loss = actor_loss.mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + critic_loss 
                writer_1.add_scalar('Loss/actor', actor_loss.item(), self.global_batch)
                writer_1.add_scalar('Loss/critic', critic_loss.item(), self.global_batch)
                self.global_batch += 1
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()
                    # Log weights and gradients of PolicyNetwork
            for name, param in self.actor.named_parameters():
                writer_1.add_histogram(f'PolicyNetwork/{name}', param, self.global_epoch)
                if param.grad is not None:
                    writer_1.add_histogram(f'PolicyNetwork/{name}.grad', param.grad, self.global_epoch)

                # Log weights and gradients of ValueNetwork
            for name, param in self.critic.named_parameters():
                writer_1.add_histogram(f'ValueNetwork/{name}', param, self.global_epoch)
                if param.grad is not None:
                    writer_1.add_histogram(f'ValueNetwork/{name}.grad', param.grad, self.global_epoch)
                
            self.global_epoch += 1  
        writer_1.close()




