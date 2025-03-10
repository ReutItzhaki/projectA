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
            #nn.Tanh(),
            #nn.Linear(64, 64),
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
        writer_1 = SummaryWriter('log_dir_0903')
        '''
        The training process consists of the following steps:
        1) Calculate advantage
        2) Calculate loss
        3) Backprop and Optimize
        4) Log performance
        5) Repeat for multiple epochs
        6) clear memory
        
        step 1: calculate advantage (A_t)
        The Advantage Function, denoted as A_t, 
        is used to assess the relative quality of a specific action compared to 
        other possible actions that could have been taken from the same state (s_t).
        If A_t > 0  → The action was better than expected based on the current policy.
        If A_t < 0  → The action was worse than expected, so its selection probability should be reduced.

        We caculate the advantage using the Generalized Advantage Estimation (GAE) method:
        A_t = ∑(γλ)^k * δ_k
        where δ_k = r_k + γV(s_k+1) - V(s_k)
        r_k is the immediate reward at time step k
        δ_k is the TD error (The difference between the current value of the state and the expected value of the state)
        γ is the discount factor
        λ is the GAE parameter- determines how much to look into the future
        '''
        for _ in range(self.n_epochs): 
            # Retrieve stored experience from memory in the form of batches
            state_arr, action_arr, old_log_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = self.memory.generate_batches()
            
            values = vals_arr # Estimated state values from the critic network = V(s_k)
            advantage = np.zeros(len(reward_arr), dtype=np.float32) # initialize the advantage array in a size of the reward array (which is the same size as the state array and the action array and more... )

            # Loop over all time steps to compute A_t
            for t in range(len(reward_arr)-1): # for each reward given
                discount = 1  # Initialize discount factor which is γλ
                a_t = 0  # Initialize advantage for time step t
                for k in range(t, len(reward_arr)-1): # Iterate over future rewards
                    if dones_arr[k]:  # Check if the state is terminal
                        #print('k=',k, 'is done')
                        break  # Stop accumulating rewards if terminal state is reached

                    # Compute the TD error (δ_k = r_k + γV(s_k+1) - V(s_k))
                    td_error = reward_arr[k] + self.gamma * values[k + 1] * (1 - int(dones_arr[k])) - values[k]
                    # 1-int(dones_arr[k]) is used to check if the state is terminal or not

                    # Accumulate advantage (A_t += (γλ)^k * δ_k)
                    a_t += discount * td_error
                    
                    # Update the discount factor (γλ)
                    discount *= self.gamma * self.gae_lambda  # discount factor is getting smaller as we look further into the future
                advantage[t] = a_t # Store computed advantage for time step t
            # Convert advantage array to tensor for GPU computation
            advantage = torch.tensor(advantage).to(self.actor.device) # convert advantage to tensor

            
            '''
            step 2: Actor-Critic Loss Computation
            there are few steps to compute the loss:
            1) PPO Clipped Loss Function (Actor Loss):
            L^{CLIP}(θ) = E_t [ min ( r_t(θ) A_t, clip (r_t(θ), 1 - ε, 1 + ε) A_t ) ]

            Where:
            ε is a small hyperparameter (typically 0.2) that prevents excessively large updates to the policy.
            r_t(θ) is the policy ratio, defined as:
            r_t(θ) = (π_θ(a_t|s_t) / π_{θ_{old}}(a_t|s_t))
            This measures the relative probability of an action under the new and old policies.
            
            this constrains the policy ratio so that the update does not exceed a predefined range, 
            thereby ensuring that the policy does not change too drastically in a single update step.

            2) Critic Loss (Value Function Update) - Implements MSE loss:
            L^{VF}(θ) = E_t [ (V_{θ}(s_t) - (V_target_t)^2 ]
            
            Where:
            V_{θ}(s_t) is the value function estimate from the critic network
            V_target_t = V_{θ_{old}}(s_t) + A_t 

            3) Total Loss:
            L(θ) = L^{CLIP}(θ) + L^{VF}(θ) - c * E_t [H(π_θ(s_t))]

            Where:
            H(π_θ(s_t)) is the entropy of the policy distribution
            c is a hyperparameter that controls the strength of the entropy term
            The entropy term encourages exploration by penalizing overly deterministic policies.
            '''
            
            values = torch.tensor(values).to(self.critic.device)
            for batch in batches:
                # Convert batch data into tensors
                states = torch.tensor(state_arr[batch], dtype=torch.float32).to(self.actor.device)
                old_log_prob = torch.tensor(old_log_prob_arr[batch], dtype=torch.float32).to(self.actor.device)
                actions = torch.tensor(action_arr[batch], dtype=torch.float32).to(self.actor.device)
                
                # Get action probabilities from the actor network
                dist = self.actor(states) # Get normal distribution (mu, 1)
                critic_value = self.critic(states) # Get state value estimate from critic = V_{θ}(s_t)
                critic_value = torch.squeeze(critic_value)  # Remove extra dimension if necessary 

                # Recompute log probability using the same transformation from choose_action
                z_n = dist.sample()  # Sample raw action from new policy
                log_det_jacobian = torch.log(1.0 - torch.tanh(z_n).pow(2))  # Log determinant correction
                new_log_prob = dist.log_prob(z_n) - log_det_jacobian  # Corrected log probability

                
                # 1) Compute PPO's clipped objective (Actor Loss)
                prob_ratio = new_log_prob.exp() / old_log_prob.exp() # computes the policy ratio =  r_t(θ) = (π_θ(a_t|s_t) / π_{θ_{old}}(a_t|s_t))
                advantage_reshaped = advantage[batch] # Extract the advantage for this batch
                weighted_probs = advantage_reshaped * prob_ratio # Unclipped probability-weighted advantage
                weighted_clipped_probs = torch.clamp(prob_ratio, 1-self.epsilon_clip,
                        1+self.epsilon_clip)*advantage_reshaped # clipping the ratio of the probabilities 
                
                # The quantity we actually want to maximize
                actor_objective = torch.min(weighted_probs, weighted_clipped_probs).mean()

                # Then define the loss we pass to the optimizer (which wants to "minimize" something):
                actor_loss = -actor_objective

                # 2) Compute Critic Loss (MSE on Value Function)
                returns = advantage[batch] + values[batch] # Compute target value V_target, value[batch] is V_{θ_{old}}(s_t)
                critic_loss = (returns-critic_value)**2 # Squared difference between estimated and target value
                critic_loss = critic_loss.mean() # Average over the batch

                # 3) Compute total loss
                entropy = dist.entropy().mean() # Entropy term for better exploration
                total_loss = actor_loss + 0.1*critic_loss - 0.01*entropy # Total loss with entropy term

                #writer_1.add_scalar('Loss/actor', actor_loss.item(), self.global_batch) # we should see the loss decreasing
                writer_1.add_scalar("ActorObjective (should go up)", actor_objective.item(), self.global_batch)
                writer_1.add_scalar("ActorLoss (should go down)", actor_loss.item(), self.global_batch)
                writer_1.add_scalar('Loss/critic', critic_loss.item(), self.global_batch)
                writer_1.add_scalar('Loss/total', total_loss.item(), self.global_batch)
                 
                '''
                step 3: Backpropagation and Optimization
                The gradients of the actor and critic networks are computed using backpropagation, with respect to total_loss.
                (The gradients are stored in each parameter's .grad attribute.)
                These gradients tell us which direction to move the weights to minimize the loss.

                after computibg the gradients, they are clipped to prevent exploding gradients.
                The weights of the networks are updated using the optimizer.

                '''
                # 1) zero the gradients of the actor and critic networks, so that the gradients do not accumulate
                self.actor.optimizer.zero_grad() 
                self.critic.optimizer.zero_grad() 

                # 2) Compute Gradients Using Backpropagation
                total_loss.backward() # compute the gradients of the total loss
                #actor_loss.backward() # compute the gradients of the actor loss
                #critic_loss.backward() # compute the gradients of the critic loss

                # 3) clip the gradients of the actor and critic networks
                #torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=50)

                # 4) Update the Weights of the Networks based on the clipped gradients
                self.actor.optimizer.step() 
                self.critic.optimizer.step() 
                
                # 5) done with the batch
                self.global_batch += 1
                '''
                log the performance of the agent during training
                1) The norm of the gradients of the actor network
                2) The norm of the gradients of the critic network
                3) The weights of the actor network
                4) The weights of the critic network
                5) The gradients of the actor network
                6) The gradients of the critic network
                '''

            # outside of the batch loop
            # instide the epoch loop    
            # Log weights and gradients of Actor Network
            for name, param in self.actor.named_parameters():
                writer_1.add_histogram(f'ActorNetwork/{name}', param, self.global_epoch) # log the weights of the actor network
                if param.grad is not None: # if the gradients are not None
                    writer_1.add_histogram(f'ActorNetwork/{name}.grad', param.grad, self.global_epoch) # log the gradients of the actor network
                    grad_norm = param.grad.norm(2).item() # Compute and log L2 norm of gradients
                    print(f"ActorNetwork/{name}: Grad Norm L2 = {grad_norm:.4f}")
                    writer_1.add_scalar(f'ActorNetwork/{name}_grad_norm', grad_norm, self.global_epoch)

            # Log weights and gradients of Value Network
            for name, param in self.critic.named_parameters():
                writer_1.add_histogram(f'ValueNetwork/{name}', param, self.global_epoch) # log the weights of the critic network
                if param.grad is not None:
                    writer_1.add_histogram(f'ValueNetwork/{name}.grad', param.grad, self.global_epoch) # log the gradients of the critic network
                    grad_norm = param.grad.norm(2).item() # Compute and log L2 norm of gradients
                    print(f"ValueNetwork/{name}: Grad Norm L2 = {grad_norm:.4f}")
                    writer_1.add_scalar(f'ValueNetwork/{name}_grad_norm', grad_norm, self.global_epoch)

            # done with the epoch
            self.global_epoch += 1
                
        writer_1.close()
        # clear memory after all epochs
        self.memory.clear_memory()



