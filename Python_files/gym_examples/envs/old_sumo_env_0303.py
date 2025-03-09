import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import traci
from traci import simulation as sim
from traci import vehicle as vcl
from sumolib import checkBinary



class SumoEnv(gym.Env):
    """
    Initializes the SumoEnv environment.
    Attributes:
        action_space (gym.spaces.Discrete): Discrete action space for the environment.
        observation_space (gym.spaces.Box): Observation space representing the state of the environment.
        control_cycle (int): Number of simulation steps to run for each action.
        render_mode (str): Mode for rendering the environment (e.g., 'sumo' or 'sumo-gui').
        traci_connected (bool): Flag to indicate if the environment is connected to SUMO.
        device (str): Device to run the environment on (e.g., 'cpu' or 'cuda').
        simulation_mode (str): Mode for running the SUMO simulation (e.g., 'sumo' or 'sumo-gui').
        step_count (int): Number of steps taken in the environment.
        sumoBinary (str): Path to the SUMO binary.
        sumo_simulation (list): List of arguments for running the SUMO simulation.
        zones (dict): Dictionary of zones in the environment.
        max_steps (int): Maximum number of steps in the environment.
        vehicle_max_count (int): Maximum number of vehicles in the environment.
        speed_max (int): Maximum speed of vehicles in the environment.
    """
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.n_observations = 8 
        #self.n_actions = 24
        # the observation space is a box with 8 values, 4 for the number of vehicles in each protected region and 4 for the avarege speed of the vehicles in each one of the protected regions
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_observations,), dtype=np.float32)
        #self.observation_space = spaces.Box(low=np.array([0.0]*28, dtype=np.float32), high=np.array([1.0]*28, dtype=np.float32), dtype=np.float32) 
        # the action space is discrete with 1 action- open or close the traffic ligths at the perimeter 
        self.action_space = spaces.Discrete(1)
        #self.action_space = spaces.MultiBinary(self.n_actions)
        self.control_cycle = 50
        self.render_mode = None 
        self.traci_connected = False
        self.device = device
        self.simulaition_mode = 'sumo' # change to 'sumo-gui' to open it graphically
        self.step_count = 0
        self.sumoBinary = checkBinary(self.simulaition_mode)
        self.sumo_simulation = [self.sumoBinary, "-c", "SUMO_files/sumoConfig2.sumocfg", "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
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
        #self.max_steps = 160 # the maximum number of steps is 160
        self.vehicle_max_count = 1200 # the maximum number of vehicles is 1200
        self.speed_max = 16 # the maximum speed is 16
        
    def count_vehicles_in_zone(self, zone_id):
        # count the number of vehicles in a zone
        count = 0
        for edges in self.zones[f"zone_{zone_id}"]:
            count += traci.edge.getLastStepVehicleNumber(edgeID=edges) + traci.edge.getLastStepVehicleNumber(edgeID=f"-{edges}")
        return count
    
    def mean_speed_of_zone(self, zone_id):
        # calculate the mean speed of vehicles in a zone
        speeds = [traci.edge.getLastStepMeanSpeed(edgeID=edge) for edge in self.zones[f"zone_{zone_id}"]]
        return np.mean(speeds)  
    
    def _getinfo(self):
        info = {
            "info": None
        }
        return info


    def reset(self, seed=None, options=None):
        # Reset the simulation to its initial state
        super().reset(seed=seed)
         # Check if there is an existing traci connection
        if self.traci_connected:
            # If so, close it before starting a new simulation
            traci.close()

        # Open a new sumo simulation
        traci.start(self.sumo_simulation)
        self.traci_connected = True
        info = self._getinfo()
        observation = np.zeros(8, dtype=np.float32)
        self.step_count = 0        
        return observation, info
    


    def step(self, action):
        terminated = False
        self.step_count += 1
        for i in range(24): # for each traffic light
            if (action == 1): # set the traffic light to red
                phase = 0
            else: # set the traffic light to green
                phase = 2
            traci.trafficlight.setPhase(f"t{i+1}", phase)

        for i in range(self.control_cycle): # Run the simulation for 50 steps
            traci.simulation.step()
            if(not sim.getMinExpectedNumber()):# Check if there are no vehicles in the simulation
                terminated = True
                break

        protected_region_densities = np.zeros(4, dtype=np.float32)
        protected_region_mean_speeds = np.zeros(4, dtype=np.float32) 
        for i in range(4): # in each protected region, i from 0 to 3 (regions 1 to 4)
            protected_region_densities[i] = self.count_vehicles_in_zone(i+1)
            if(protected_region_densities[i] > 1200):
                protected_region_densities[i] = 1200

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
        # the reward is the sum of the vehicle count penalty and the speed penalty
        reward = vehicle_count_penalty + speed_penalty
        
        #if (self.step_count == self.max_steps):
        #    terminated = True

        return observation, reward, terminated, False, info

    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        return super().close()

from gymnasium.envs.registration import register

register(
     id="gym_examples/Sumo",
     entry_point="gym_examples.envs:SumoEnv",
     max_episode_steps=160
)
