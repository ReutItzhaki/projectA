import gym
from gym import spaces, Wrapper
import numpy as np
import traci  # TraCI library for SUMO
from traci import simulation as sim 
from traci import vehicle as vcl
from sumolib import checkBinary
from torchrl.envs.utils import check_env_specs

# register the environment
gym.envs.register(
    id='SumoEnv-v0',
    entry_point='sumo_env:SumoEnv',
    max_episode_steps=160,
)


class SumoEnv(gym.Env):
    def __init__(self):
        """
        Initializes the SumoEnv environment.
        Parameters:
            sumo_config_file (str): Path to the SUMO configuration file.
        Attributes:
            sumo_config_file (str): Stores the path to the SUMO configuration file.
            sumo_cmd (list): Command to start the SUMO simulation.
            action_space (gym.spaces.Discrete): Discrete action space for the environment.
            observation_space (gym.spaces.Box): Observation space representing the state of the environment.
        The `super(SumoEnv, self).__init__()` call initializes the parent class of SumoEnv.
        """
        super(SumoEnv, self).__init__()        
        # Initialize SUMO
        self.sumo_config_file = "kaggle_081224/sim5.sumocfg"
        #self.sumo_config_file = sumo_config_file # this is the path to the sumo configuration file
        self.sumo_cmd = ["sumo", "-c", self.sumo_config_file] # this is the command to start the sumo simulation
        self.action_space = gym.spaces.Discrete(1) # the action space is discrete with 1 action- open or close the traffic ligths at the perimeter 
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(8,), dtype=np.float32) # the observation space is a box with 8 values, 4 for the number of vehicles in each protected region and 4 for the avarege speed of the vehicles in each one of the protected region
        # initialize all the parameter for the environment
        self.simulaition_mode = 'sumo' # change to 'sumo-gui' to open it graphically
        self.traci_connected = False # we are not connected to traci yet
        self.control_cycle = 50 # the control cycle is 50 seconds
        self.step_count = 0 # the step count is 0
        self.sumoBinary = checkBinary(self.simulaition_mode)
        self.sumo_simulation = [self.sumoBinary, "-c", self.sumo_config_file, "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        self.device = 'cpu' # sumo is running on the cpu
        self.rander_mode = None # there is no render mode
        self.zones = {
                # up-left
                "zone_1": ("E0", "E1", "E7", "E8", "E383", "E321", "E301", "E302", "d1", "d2", "d3", "d4",
                        "E374", "E375", "E299", "E300", "E322", "E385", "E298", "E376", "d5", "d6", "d7", "d8",
                        "d9", "d10", "d12", "E303", "E304", "E319", "E320", "E292", "E293", "E290", "E291", 
                        "E324", "E323", "E289", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
                        "d29", "d30", "d32", "E318", "E325", "E305", "d41", "d42", "d43", "d45", "d46", "d47",
                        "d49", "d50", "E9", "E11", "E13", "E54", "E52"),
                # up-right           
                "zone_2": ("E377", "E297", "E341", "E387", "d11", "E378", "E379", "E3", "E296", "E380",
                        "E381", "E4", "E345", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
                        "E347", "E391", "d31", "E288", "E339", "E340", "E286", "E287", "E6", "E68", "E366", "E367",
                        "E348", "E349", "d33", "d34", "d35", "d36", "d37", "d38", "d39", "d40", "E338", "E69", "E350",
                        "d51", "d53", "d54", "d55", "d57", "d58", "d59", "E50", "E48", "E46", "E38", "E40", "E44"), 
                # down-right
                "zone_3": ("E351", "E352", "E353", "E354", "E355", "E70", "E71", "E72", "E73", "E74",
                        "E333", "E334", "E335", "E336", "E337", "E278", "E279", "E280","E363", "E364",
                        "E144", "E138", "E139", "E360", "E361", "E76", "E77", "E78", "E357", "E358",
                        "d91", "d93", "d94", "d95", "d96", "d97", "d98", "d99", "d100", 
                        "d71", "d73", "d74", "d75", "d76", "d77", "d78", "d79", "d80", 
                        "d51", "d54", "d55", "d56", "d57", "d58", "d59", "d60", "E34", "E36", "E42", "E28", "E30", "E32"),
                # down-left
                "zone_4": ("E306", "E307", "E308", "E309", "E310", "E313", "E314", "E315", "E316", "E317",
                        "E326", "E327", "E328", "E329", "E330", "E281", "E282", "E283", "E284", "E285",
                            "E273", "E274", "E275", "E276", "E277", "E135", "E136", "E137", "E79", "E80",
                            "d44", "d48", "d52", "d61", "d62", "d63", "d64", "d65", "d66", "d67", "d68", "d69", 
                            "d81", "d82", "d83", "d84", "d85", "d86", "d87", "d88", "d89", "E22", "E24", "E26",
                            "E16", "E18", "E20")
        }
        self.max_steps = 160 # the maximum number of steps is 160
        self.vehicle_max_count = 1200 # the maximum number of vehicles is 1200
        self.speed_max = 16 # the maximum speed is 16
    def reset(self):
        # Reset the simulation to its initial state
        super(SumoEnv, self).reset() # Reset the parent class
        traci.start(self.sumo_simulation) # Start the SUMO simulation
        self.traci_connected = True # Set the flag to indicate that we are connected to SUMO
        observation = np.zeros(8) # initialize the observation to zeros
        self.step_count = 0 # Reset the step count
        info = self._getinfo() # Get the information about the environment 

        return observation, info # Return the initial observation and the information
    
    # count the number of vehicles in each zone
    def count_vehicles_in_zone(self, zone_id):
        count = 0
        for edges in self.zones[f"zone_{zone_id}"]:
            count += traci.edge.getLastStepVehicleNumber(edgeID=edges) + traci.edge.getLastStepVehicleNumber(edgeID=f"-{edges}")
        return count
    
    def mean_speed_of_zone(self, zone_id):
        # Batch fetching and computing mean speed
        speeds = [traci.edge.getLastStepMeanSpeed(edgeID=edge) for edge in self.zones[f"zone_{zone_id}"]]
        return np.mean(speeds)  
     
    def step(self, action):
        self.apply_action(action) # Apply action (e.g., change traffic light)
        terminated = False
        self.step_count += 1

        for i in range(self.control_cycle): # Run the simulation for 50 steps
            traci.simulation.step()
            if(not sim.getMinExpectedNumber()): # Check if there are no vehicles in the simulation
                terminated = True
                break

        # Compute the observation
        protected_region_densities = np.zeros(4, dtype=np.float32) # Initialize the protected region densities
        protected_region_mean_speeds = np.zeros(4, dtype=np.float32) # Initialize the protected region mean speeds

        for i in range(4): # Compute the number of vehicles in each protected region
            
            protected_region_densities[i] = self.count_vehicles_in_zone(i+1)
            if(protected_region_densities[i] > self.vehicle_max_count):
                protected_region_densities[i] = self.vehicle_max_count
            
            protected_region_mean_speeds[i] = self.mean_speed_of_zone(i+1)
            if(protected_region_mean_speeds[i] > self.speed_max):
                protected_region_mean_speeds[i] = self.speed_max

        protected_region_densities = protected_region_densities / self.vehicle_max_count # Normalize the densities
        protected_region_mean_speeds = protected_region_mean_speeds / self.speed_max # Normalize the speeds

        info = self._getinfo()
        observation = np.concatenate([protected_region_densities, protected_region_mean_speeds])

        # Penalize higher vehicle count
        vehicle_count_penalty = -(vcl.getIDCount()) / 5000.0
        # give higher reward for higher speed
        speed_penalty = np.sum(protected_region_mean_speeds) / 4
        reward = vehicle_count_penalty + speed_penalty

        if (self.step_count == self.max_steps): # Check if the simulation has run for the maximum number of steps
            terminated = True

        return observation, reward, terminated, False, info
        
    
    def apply_action(self, action):
        # Implement the logic to apply actions, e.g., changing traffic light phases
        if action == 1:
            # Set traffic light to red
            phase = 0 # Set the phase to red
        else: # Set the phase to green
            phase = 2 
        
        for i in range(24): # 24 traffic lights
            traci.trafficlight.setPhase(f"t{i+1}", phase) # Set the phase of the traffic light

        

    def _getinfo(self):
        # Get the information about the environment
        info = {
            "info": None
        }
        return info
    
    def close(self):
        # Close the SUMO simulation
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        return super().close()
