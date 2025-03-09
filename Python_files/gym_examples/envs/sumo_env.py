import numpy as np 
import gymnasium as gym
from gymnasium import spaces
import sumolib
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
    def __init__(self, sumoConfig, device = 'cpu'):
        super().__init__()
        # Observation space: 2 variables (Dp, Df) each in [0.0..1.0].
        self.n_observations = 2 
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_observations,), dtype=np.float32)
        # Action space: 1D continuous in [50..300].
        # The array shape is (1,) to match the output of the policy network.
        self.n_actions = 1
        self.action_space = spaces.Box(
            low=np.array([50.0], dtype=np.float32),
            high=np.array([300.0], dtype=np.float32),
            shape=(self.n_actions,),
            dtype=np.float32
        )
        # SUMO execution parameters
        self.simulaition_mode = 'sumo' # change to 'sumo-gui' to open it graphically
        self.sumoBinary = checkBinary(self.simulaition_mode)
        # init value, will be updated in reset()
        #sumoConfig = "SUMO_files/sumoConfig2.sumocfg"
        self.sumo_simulation = [self.sumoBinary, "-c", sumoConfig, "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        #self.sumo_simulation = [self.sumoBinary, "-c", "SUMO_files/sumoConfig2.sumocfg", "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        #self.sumo_simulation=[self.sumoBinary, "-c", "SUMO_files/sim0503.sumocfg", "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        print("init sumo env")
        self.traci_connected = False
        self.control_cycle = 50
        self.render_mode = None 
        self.device = device
        self.step_count = 0
        self.protected_edges = ["E0", "E1", "E7", "E8", "E383", "E321", "E301", "E302", "d1", "d2", "d3", "d4",
                        "E374", "E375", "E299", "E300", "E322", "E385", "E298", "E376", "d5", "d6", "d7", "d8",
                        "d9", "d10", "d12", "E303", "E304", "E319", "E320", "E292", "E293", "E290", "E291", 
                        "E324", "E323", "E289", "d21", "d22", "d23", "d24", "d25", "d26", "d27", "d28",
                        "d29", "d30", "d32", "E318", "E325", "E305", "d41", "d42", "d43", "d45", "d46", "d47",
                        "d49", "d50", "E9", "E11", "E13", "E54", "E52",
                        "E377", "E297", "E341", "E387", "d11", "E378", "E379", "E3", "E296", "E380",
                        "E381", "E4", "E345", "d13", "d14", "d15", "d16", "d17", "d18", "d19", "d20",
                        "E347", "E391", "d31", "E288", "E339", "E340", "E286", "E287", "E6", "E68", "E366", "E367",
                        "E348", "E349", "d33", "d34", "d35", "d36", "d37", "d38", "d39", "d40", "E338", "E69", "E350",
                        "d51", "d53", "d54", "d55", "d57", "d58", "d59", "E50", "E48", "E46", "E38", "E40", "E44",
                        "E351", "E352", "E353", "E354", "E355", "E70", "E71", "E72", "E73", "E74",
                        "E333", "E334", "E335", "E336", "E337", "E278", "E279", "E280","E363", "E364",
                        "E144", "E138", "E139", "E360", "E361", "E76", "E77", "E78", "E357", "E358",
                        "d91", "d93", "d94", "d95", "d96", "d97", "d98", "d99", "d100", 
                        "d71", "d73", "d74", "d75", "d76", "d77", "d78", "d79", "d80", 
                        "d51", "d54", "d55", "d56", "d57", "d58", "d59", "d60", "E34", "E36", "E42", "E28", "E30", "E32",
                        "E306", "E307", "E308", "E309", "E310", "E313", "E314", "E315", "E316", "E317",
                        "E326", "E327", "E328", "E329", "E330", "E281", "E282", "E283", "E284", "E285",
                            "E273", "E274", "E275", "E276", "E277", "E135", "E136", "E137", "E79", "E80",
                            "d44", "d48", "d52", "d61", "d62", "d63", "d64", "d65", "d66", "d67", "d68", "d69", 
                            "d81", "d82", "d83", "d84", "d85", "d86", "d87", "d88", "d89", "E22", "E24", "E26",
                            "E16", "E18", "E20"]
        
        self.feeder_edges = ["f1", "f2", "f3", "f4","f5", "f6", "f7", "f8", "f9", "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19", "f20", "f21", "f22", "f23", "f24"]

        # Read Edge Lengths from the net.xml
        net_file_path = "SUMO_files/network.net.xml"
        net = sumolib.net.readNet(net_file_path)

        self.edge_lengths = {}
        for edge_obj in net.getEdges():
            edge_id = edge_obj.getID()
            self.edge_lengths[edge_id] = edge_obj.getLength()  # length in meters

        
    
    def set_perimeter_meter(self, flow_rate):
        """
        Convert the chosen flow_rate (vehicles/hour) into
        a traffic light plan that enforces that approximate rate.
        In reality, you'll want a more accurate model of
        capacity vs green time, but here's a simple demo.
        """
        max_flow_rate = 300.0
        min_flow_rate = 50.0

        # Map flow_rate to [0..1], then to [0..cycle_length] for green time.
        ratio = (flow_rate - min_flow_rate) / (max_flow_rate - min_flow_rate)
        green_time = ratio * self.control_cycle
        red_time = self.control_cycle - green_time
        int_green = max(1, int(green_time))
        int_red   = max(1, int(red_time))

        
        logic = traci.trafficlight.Logic(
            programID="0",
            type=0,  # static
            currentPhaseIndex=0,  # or another int
            #subParameter=None,
            #offset="0",
            phases=[
                traci.trafficlight.Phase(
                    duration=int(int_green),
                    #minDur=int(green_time),
                    #maxDur=int(green_time),
                    state="GGGG"  # Green
                )
                ,
                traci.trafficlight.Phase(
                    duration=0,
                    #minDur=0,
                    #maxDur=0,
                    state="yyyy"  # Yellow
                )
                ,
                traci.trafficlight.Phase(
                    duration=int(int_red),
                    #minDur=int(red_time),
                    #maxDur=int(red_time),
                    state="rrrr"  # Red
                )
            ]
        )
        for i in range(24):
            traci.trafficlight.setProgramLogic("t"+str(i+1), logic)
            #traci.trafficlight.setCompleteRedYellowGreenDefinition(f"t{i+1}", logic)
    
    def _compute_density(self, edges):
        """
        Implements D = sum(V_i) / sum(L_i),
        summing over all edges in the given list.
        If a reverse-edge "-edge" exists in the network, we also include it.
        """
        total_vehicles = 0.0
        total_length = 0.0

        for edge in edges:
            # 1) Vehicles on this edge
            v_edge = traci.edge.getLastStepVehicleNumber(edge)
            # 2) Edge length
            L_edge = self.edge_lengths.get(edge, 0.0)

            total_vehicles += v_edge
            total_length += L_edge

            # Check if there's a reverse edge
            rev_edge = f"-{edge}"
            if rev_edge in self.edge_lengths:
                v_rev = traci.edge.getLastStepVehicleNumber(rev_edge)
                L_rev = self.edge_lengths[rev_edge]
                total_vehicles += v_rev
                total_length += L_rev

        if total_length > 0:
            return total_vehicles / total_length
        else:
            return 0.0
        

    def _getinfo(self):
        info = {
            "info": None
        }
        return info


    def reset(self ,seed=None, options=None):
        # Reset the simulation to its initial state
        super().reset(seed=seed)
         # Check if there is an existing traci connection
        if self.traci_connected:
            # If so, close it before starting a new simulation
            traci.close()
        print("before start")
        # Open a new sumo simulation
        traci.start(self.sumo_simulation)
        print("after start")
        #traci.start(sumoConfig)
        self.traci_connected = True
        info = self._getinfo()
        observation = np.zeros(self.n_observations, dtype=np.float32)
        self.step_count = 0        
        return observation, info
    


    def step(self, action):
        """
        The action is a 1D array (shape=(1,)) in [50..300] representing flow_rate (veh/h).
        We'll set the perimeter meter, run SUMO for 'control_cycle' steps,
        then compute the trip-completion-rate as reward,
        and recompute [Dp, Df] from the sums of vehicles / lengths.
        """
        flow_rate = float(action[0])
        flow_rate = np.clip(flow_rate, 50.0, 300.0)
        self.set_perimeter_meter(flow_rate)

        arrived_vehicles = 0
        terminated = False

        for _ in range(self.control_cycle):
            traci.simulation.step()
            arrived_vehicles += traci.simulation.getArrivedNumber()
            #print("arrived_vehicles: ", traci.simulation.getArrivedNumber())
            #print("expected_vehicles: ", sim.getMinExpectedNumber())
            # print the number of cars in the simulation that are not waiting
            #print("not waiting: ", vcl.getIDCount())
            if not sim.getMinExpectedNumber():
                # no more vehicles expected => we can terminate
                print("no more vehicles expected")
                terminated = True
                break

        # Reward = (#arrived over control_cycle) / (control_cycle seconds)
        # If we broke early, we still just use control_cycle to keep it simpler,
        # but you could also adjust to the actual # of steps done.

        #calculte the mean of waiting timefor one vehicle in the feeder edges
        total_mean_waiting_time = 0
        for edge in self.feeder_edges:
            waiting_time = traci.edge.getWaitingTime(edge)
            #how many vehicles are waiting in the edge
            vehicles_waiting = traci.edge.getLastStepHaltingNumber(edge)
            if(vehicles_waiting != 0):
                waiting_time /= vehicles_waiting
            else:
                waiting_time = 0

            total_mean_waiting_time += waiting_time
        total_mean_waiting_time /= len(self.feeder_edges)

        total_mean_waiting_time_2 = 0
        for edge in self.protected_edges:
            waiting_time = traci.edge.getWaitingTime(edge)
            #how many vehicles are waiting in the edge
            vehicles_waiting = traci.edge.getLastStepHaltingNumber(edge)
            if(vehicles_waiting != 0):
                waiting_time /= vehicles_waiting
            else:
                waiting_time = 0

            total_mean_waiting_time_2 += waiting_time
        total_mean_waiting_time_2 /= len(self.protected_edges)

        total_mean_waiting_time = (total_mean_waiting_time + total_mean_waiting_time_2) /1000

        #reward= -total_mean_waiting_time
        #print("arrived_vehicles: ", arrived_vehicles)
        #print("min expected number: ", sim.getMinExpectedNumber())
        reward = arrived_vehicles / float(self.control_cycle)
        #print("reward: ", reward)
        #print("arrived_vehicles: ", arrived_vehicles)

        # Now compute densities according to the formula:
        # Dp = sum(V_i)/sum(L_i) for i in protected edges
        # Df = sum(V_i)/sum(L_i) for i in feeder edges
        Dp = self._compute_density(self.protected_edges)
        Df = self._compute_density(self.feeder_edges)

        obs = np.array([Dp, Df], dtype=np.float32)
        self.step_count += 1

        info = self._getinfo()
        return obs, reward, terminated, False, info

    def close(self):
        if self.traci_connected:
            traci.close()
            self.traci_connected = False
        return super().close()

from gymnasium.envs.registration import register

register(
     id="gym_examples/Sumo",
     entry_point="gym_examples.envs:SumoEnv",
     max_episode_steps=60
)
