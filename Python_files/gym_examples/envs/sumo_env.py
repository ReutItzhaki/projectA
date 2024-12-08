import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import traci
from traci import simulation as sim
from traci import vehicle as vcl
from sumolib import checkBinary


class SumoEnv(gym.Env):
    
    def __init__(self, device = 'cpu'):
        super().__init__()
        self.n_observations = 28
        self.n_actions = 24
        self.observation_space = spaces.Box(low=np.array([0.0]*28, dtype=np.float32), high=np.array([1.0]*28, dtype=np.float32), dtype=np.float32) 
        self.action_space = spaces.MultiBinary(self.n_actions)
        self.control_cycle = 50
        self.render_mode = None 
        self.traci_connected = False
        self.device = device
        self.simulaition_mode = 'sumo' # change to 'sumo-gui' to open it graphically
        self.step_count = 0
        self.sumoBinary = checkBinary(self.simulaition_mode)
        self.sumo_simulation = [self.sumoBinary, "-c", "SUMO_files/sim5.sumocfg", "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
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
        
    def count_vehicles_in_zone(self, zone_id):
        count = 0
        for edges in self.zones[f"zone_{zone_id}"]:
            count += traci.edge.getLastStepVehicleNumber(edgeID=edges) + traci.edge.getLastStepVehicleNumber(edgeID=f"-{edges}")
        return count
    def _getinfo(self):
        info = {
            "info": None
        }
        return info


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
         # Check if there is an existing traci connection
        if self.traci_connected:
            # If so, close it before starting a new simulation
            traci.close()

        # Open a new sumo simulation
        traci.start(self.sumo_simulation)
        self.traci_connected = True
        info = self._getinfo()
        observation = np.zeros(28, dtype=np.float32)
        self.step_count = 0        
        return observation, info
    


    def step(self, action):

        terminated = False
        self.step_count += 1
        for i in range(24):
            if (action[i] == 1):
                phase = 0
            else:
                phase = 2
            traci.trafficlight.setPhase(f"t{i+1}", phase)

        for i in range(self.control_cycle):
            traci.simulation.step()
            if(not sim.getMinExpectedNumber()):
                terminated = True
                break

        feeder_densities = np.zeros(24, dtype=np.float32)
        for i in range (24):
            feeder_densities[i] = traci.edge.getLastStepVehicleNumber(edgeID=f"f{i+1}")
            if(feeder_densities[i] > 800):
                feeder_densities[i] = 800    

        protected_region_densities = np.zeros(4, dtype=np.float32)
        for i in range(4):
            protected_region_densities[i] = self.count_vehicles_in_zone(i+1)
            if(protected_region_densities[i] > 1200):
                protected_region_densities[i] = 1200

        feeder_densities = feeder_densities / 800
        protected_region_densities = protected_region_densities / 1200

        info = self._getinfo()
        observation = np.concatenate([protected_region_densities, feeder_densities])
        reward = -(vcl.getIDCount())
        reward = float(reward)
        reward /= 5000        
        if (self.step_count == 160):
            terminated = True

        return observation, reward, terminated, False, info

    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        return super().close()

# def render???

from gymnasium.envs.registration import register

register(
     id="gym_examples/Sumo",
     entry_point="gym_examples.envs:SumoEnv",
     max_episode_steps=160
)
