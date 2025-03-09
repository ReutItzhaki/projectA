import os
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.distributions.bernoulli import Bernoulli
from torch.distributions.normal import Normal
import torch.nn.init as init


from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here

############################# Data Store ####################################################
class Memory:
    """
    A class to store and manage experience replay memory.
    operates as a FIFO - when reaching full capacity the first sample is overriden.
    Attributes:
    states (list): The states of the environment (the observations from the environment).
    probs (list): The probabilities of the actions taken (the output of the actor network).
    vals (list): The values of the states (the output of the critic network).
    actions (list): The actions taken.
    rewards (list): The rewards received.
    dones (list): The done signals (for each step in the simulation: true if the simulation is done, false otherwise).
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
        ## suppose n_states=20 and batch_size = 4, there are 5 batches
        n_states = len(self.states)
        ##n_states should be always greater than batch_size
        ## batch_start is an array with the starting index of every batch
        ## eg:   array([ 0,  4,  8, 12, 16]))
        batch_start = np.arange(0, n_states, self.batch_size) # creating an array with n_states/batch_size elements, the size of each batch is batch_size
        ## random shuffling indexes of the states
        # eg: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
        indices = np.arange(n_states, dtype=np.int64) # creating an array with n_states elements
        ## eg: array([12, 17,  6,  7, 10, 11, 15, 13, 18,  9,  8,  4,  3,  0,  2,  5, 14,19,  1, 16])
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        ## eg: [array([12, 17,  6,  7]),array([10, 11, 15, 13]),array([18,  9,  8,  4]),array([3, 0, 2, 5]),array([14, 19,  1, 16])]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches # 2D array with the indexes of each batch: [[batch1],[batch2],...,[batchN]]

    def store_memory(self, state, action, probs, vals, reward, done):
        #if there is not enough memory, remove the first element
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

############################ Actor Network ######################################
class ActorNwk(nn.Module):
    """
    The actor neural network.

    Attributes:
    checkpoint_file (str): Path to save/load the model.
    actor: The actor network architecture which consists of 2 hidden layers with ReLU activation functions.
    optimizer: The optimizer used for gradient descent.
    device (torch.device): The device for calculations (CPU/GPU).
    """
    def __init__(self, n_actions, n_observations, lr, save_dir, num_cells=64):
        super(ActorNwk, self).__init__()

        self.checkpoint_file = os.path.join(save_dir, 'policy_net15.pth')
        self.actor = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1) # output is a single number
        )
        # initialize the weights of the network
        for m in self.actor:
            if isinstance(m, nn.Linear): # if the layer is a linear layer
                init.xavier_normal_(m.weight, gain=1) # initialize the weights of the layer with xavier normal initialization
                init.constant_(m.bias, 0)
    
        

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr) 
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state_tensor):
        """
        Returns a Normal distribution with mean=mu, std=1.
        We'll transform the *sample* later.
        """
        mu = self.actor(state_tensor)   # shape [batch,1], can be any real number
        dist = Normal(mu, 1.0)         # fixed std=1
        return dist

    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_state_dict(torch.load(self.checkpoint_file, map_location=device))

############################### Crirtic Network ######################################
class CriticNwk(nn.Module):
    """
    Value/Critic neural network.
    The critic network purpose is to estimate the value of the state, 
    which helps the agent understand how good a given state is in terms of expected future rewards.
    Attributes:
    checkpoint_file (str): Path to save/load the model.
    critic: The value/critic network architecture: 2 hidden layers with ReLU activation functions.
    optimizer: The optimizer used for gradient descent.
    device (torch.device): The device for calculations (CPU/GPU).
    """
    def __init__(self, n_observations, lr, save_dir, num_cells=64):
        super(CriticNwk, self).__init__()

        self.checkpoint_file = os.path.join(save_dir, 'value_net15.pth')
        self.critic = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)  # Value output
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

############################# Agent ########################################3
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
    actor (ActorNwk): The policy network.
    critic (CriticNwk): The value network.
    memory (Memory): The experience replay memory.

    Methods:
    remember(observation, action, vals, reward, done): Stores data in memory.
    save_models(): Saves the actor and critic models.
    load_models(): Loads the actor and critic models.
    choose_action(observation): Chooses an action based on the observation.
    learn(): Trains the agent using stored memory.
    """
    def __init__(self, n_actions, n_observations, save_dir, gamma=0.9, lr=0.01, gae_lambda=0.95,
            epsilon_clip=0.2, batch_size=64, n_epochs=10, capacity=5e5):
        self.gamma = gamma
        self.lr=lr
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda
        self.epsilon_clip = epsilon_clip
        self.global_epoch = 0
        self.global_batch = 0
        self.n_actions = n_actions
        self.capacity = capacity
        self.n_observations = n_observations

        self.actor = ActorNwk(n_actions, n_observations, lr, save_dir=save_dir)
        self.critic = CriticNwk(n_observations, lr, save_dir=save_dir)
        self.memory = Memory(batch_size, capacity)
       
    def remember(self, observation, action, probs, vals, reward, done):
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
        """
        1) forward -> Normal(mu, 1)
        2) sample raw_x
        3) transform: a = 50 + 250 * sigmoid(raw_x)
        => final action in [50,300]
        4) BUT we do not fix log_prob => mild mismatch
        """
        
        state = torch.tensor(observation, dtype=torch.float32).to(self.actor.device)
        
        # 1) Dist = Normal(mu, 1)
        dist = self.actor(state)  
        
        # 2) sample raw_x from that Normal
        z_n = dist.sample()     # shape []
        z= torch.sigmoid(z_n)

        # 3) transform => final action in [50..300]
        #    a = 50 + 250 * sigmoid(raw_x)
        action = 50.0 + 250.0 * z
        print('action:', action)
        log_det_jacobian = torch.log(1.0 - torch.tanh(z_n).pow(2))
        log_p_z = dist.log_prob(z_n) - log_det_jacobian
        
        # Critic for state value
        value = self.critic(state).item()
        
        # Convert to numpy
        action_np = action.detach().cpu().numpy()
        log_prob_np = log_p_z.detach().cpu().numpy()
        
        return action_np, log_prob_np, value


    def learn(self):
        print('learn')
        writer_1 = SummaryWriter('log_dir_0303')

        ###### calculate advantage for each optional action (considering the future rewards) ######
        for _ in range(self.n_epochs): 
            #for each epoch, we will go through the entire memory, and devide it into batches
            # we will calculate the advantage for each batch
            # and then calculate the actor and critic loss for each batch
            # and then update the weights of the actor and critic networks
            # thats in order to ansure that the agent is learning from the entire memory
            
            state_arr, action_arr, old_log_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            for t in range(len(reward_arr)-1): # for each reward given
                discount = 1
                a_t = 0 
                for k in range(t, len(reward_arr)-1): # going through the future rewards, relatively to the current reward
                    if dones_arr[k]:  # Check if the state is terminal
                        break  # Do not include value of next state if current state is terminal
                    a_t += discount*(reward_arr[k] + self.gamma*values[k+1]*\
                            (1-int(dones_arr[k])) - values[k]) # calculate advantage for k time step (future rewards)
                    discount *= self.gamma*self.gae_lambda # discount factor is getting smaller for each future reward by gamma*lambda
                advantage[t] = a_t
            advantage = torch.tensor(advantage).to(self.actor.device) # convert advantage to tensor

            

            values = torch.tensor(values).to(self.critic.device)
            for batch in batches:
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_log_probs = torch.tensor(old_log_prob_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)
                
                dist = self.actor(states) # get the distribution of the actions from the actor network (the output of the network)
                critic_value = self.critic(states) # get the value from the critic network

                critic_value = torch.squeeze(critic_value) # remove the extra dimension
            
                # probability transformation Amit changed it is not correct yet
                z_n = dist.sample(actions)
                z = torch.sigmoid(z_n)
                log_det_jacobian = torch.log(1.0 - torch.tanh(z_n).pow(2))
                log_p_z = dist.log_prob(z_n) - log_det_jacobian
                new_log_probs = log_p_z 

                #new_log_probs = dist.log_prob(actions) # calculate the log probability of the actions taken (based on our sample in choose_action)
                
                new_log_prob = torch.sum(new_log_probs, dim=1) # sum the log probabilities of the actions taken
                old_log_prob = torch.sum(old_log_probs, dim=0) # used to be dim=1
                prob_ratio = new_log_prob.exp() / old_log_prob.exp() # vector of the ratio of the new log probability to the old log probability

                advantage_reshaped = advantage[batch] # taking the advantage of the batch
                weighted_probs = advantage_reshaped * prob_ratio 
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.epsilon_clip,
                        1+self.epsilon_clip)*advantage_reshaped # clipping the ratio of the probabilities
                
                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs) 
                # averaging over the entire current batch
                actor_loss = actor_loss.mean() # calculate the mean of the actor loss (over the batch)

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2 # calculate the critic loss which is the difference between the returns and the critic value in squared
                critic_loss = critic_loss.mean() # calculate the mean of the critic loss (over the batch)

                total_loss = actor_loss + critic_loss 
                writer_1.add_scalar('Loss/actor', actor_loss.item(), self.global_batch)
                writer_1.add_scalar('Loss/critic', critic_loss.item(), self.global_batch)
                self.global_batch += 1 
                self.actor.optimizer.zero_grad() # zero the gradients of the actor network, so that the gradients do not accumulate
                self.critic.optimizer.zero_grad() # zero the gradients of the critic network, so that the gradients do not accumulate
                total_loss.backward() # compute the gradients of the total loss
                # clip the gradients of the actor network
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
                # clip the gradients of the critic network
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=50)
                self.actor.optimizer.step() # update the weights of the actor network
                self.critic.optimizer.step() # update the weights of the critic network
                #print the reward of the batch
                #writer_1.add_scalar('Performance/Reward', np.sum(reward_arr[batch]), self.global_batch)
                #print the mean reward of the batch
                writer_1.add_scalar('Performance/MeanReward', np.mean(reward_arr[batch]), self.global_batch)
                #print the norm of the gradients of the actor network
                #writer_1.add_scalar('Gradients/actor', torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1), self.global_batch)
            # Log weights and gradients of PolicyNetwork
            for name, param in self.actor.named_parameters():
                writer_1.add_histogram(f'PolicyNetwork/{name}', param, self.global_epoch)
                if param.grad is not None: # if the gradients are not None
                    writer_1.add_histogram(f'PolicyNetwork/{name}.grad', param.grad, self.global_epoch)
                    #print the norm of the gradients of the actor network
                    #writer_1.add_scalar(f'norm2-Actor (try1)/{name}', torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1), self.global_epoch)
                    #try2
                    #grad_norm = param.grad.norm(2).item()
                    #writer_1.add_scalar(f"'norm2-Actor (try2)'/{name}", grad_norm)
                    
                    # Compute and log L2 norm of gradients
                    grad_norm = param.grad.norm(2).item()
                    print(f"PolicyNetwork/{name}: Grad Norm L2 = {grad_norm:.4f}")
                    writer_1.add_scalar(f'PolicyNetwork/{name}_grad_norm', grad_norm, self.global_epoch)

            # Log weights and gradients of ValueNetwork
            for name, param in self.critic.named_parameters():
                writer_1.add_histogram(f'ValueNetwork/{name}', param, self.global_epoch)
                if param.grad is not None:
                    writer_1.add_histogram(f'ValueNetwork/{name}.grad', param.grad, self.global_epoch)
                     #print the norm of the gradients of the actor network
                    #writer_1.add_scalar(f'norm2-Critic (try1)/{name}', torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1), self.global_epoch)
                    #try2
                    #grad_norm = param.grad.norm(2).item()
                    #writer_1.add_scalar(f"'norm2-Critic (try2)'/{name}", grad_norm)

                     # Compute and log L2 norm of gradients
                    grad_norm = param.grad.norm(2).item()
                    print(f"ValueNetwork/{name}: Grad Norm L2 = {grad_norm:.4f}")
                    writer_1.add_scalar(f'ValueNetwork/{name}_grad_norm', grad_norm, self.global_epoch)
            self.global_epoch += 1  
        writer_1.close()




