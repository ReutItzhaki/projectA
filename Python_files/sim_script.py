import traci
import traci.constants as tc
from traci import simulation as sim
from traci import polygon
from traci import vehicle as vcl
from traci import edge
from sumolib import checkBinary
import numpy as np
# import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
import time
import heatmap

# simulation configuration
simulaition_mode = 'sumo'
sumoBinary = checkBinary(simulaition_mode)
sumo_simulation = [sumoBinary, "-c", "SUMO_files/sim3107num2.sumocfg", "--no-warnings", "--no-step-log", "--device.rerouting.threads", "2"]  # Add --no-warnings option

# open sumo
traci.start(sumo_simulation)

# # initialize which information we want to extract
# zones = ("zone_11", "zone_12", "zone_13", "zone_14", "zone_15",
#           "zone_21", "zone_22", "zone_23", "zone_24", "zone_25",
#          "zone_31", "zone_32", "zone_33", "zone_34", "zone_35",
#          "zone_41", "zone_42", "zone_43", "zone_44", "zone_45",
#          "zone_51", "zone_52", "zone_53", "zone_54", "zone_55"
#          )
# polygon.subscribeContext('protected_region', tc.CMD_GET_VEHICLE_VARIABLE, 90)
# polygon.subscribe()
# for zone in zones:
#     polygon.subscribeContext(zone, tc.CMD_GET_VEHICLE_VARIABLE, 90)


zones = {
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

def count_vehicles_in_zone(zone_id):
    count = 0
    for edges in zones[f"zone_{zone_id}"]:
        count += edge.getLastStepVehicleNumber(edgeID=edges) + edge.getLastStepVehicleNumber(edgeID=f"-{edges}")
    return count

    




sim_time_step = np.array([])  # Initialize as empty NumPy arrays
running_vehicles = np.array([])
vehicles_in_protected_region = np.array([])
arrived_vehicles_in_cycle = np.array([])

arrived_in_cycle = 0
step = 0
cycle = 0
control_cycle_time = 100
vehicles_in_zones = np.zeros(shape=(2,2))
matrix_list = []

# begin simulation
while sim.getMinExpectedNumber():
    traci.simulationStep()
    if step == 0:
        start_cycle_time = start_time = time.time()

    arrived_in_cycle += sim.getArrivedNumber()

    # extracting information from SUMO each control cycle
    if step % control_cycle_time == 0:
        vehicles_in_zones[0][0] = count_vehicles_in_zone(1)
        vehicles_in_zones[0][1] = count_vehicles_in_zone(2)
        vehicles_in_zones[1][1] = count_vehicles_in_zone(3)
        vehicles_in_zones[1][0] = count_vehicles_in_zone(4)
        vehicles_in_protected_region = np.append(vehicles_in_protected_region, np.sum(vehicles_in_zones))     

        matrix_list.append(vehicles_in_zones.copy())
        
        
                                    



        running_vehicles = np.append(running_vehicles, vcl.getIDCount())
        sim_time_step = np.append(sim_time_step, sim.getTime())
        arrived_vehicles_in_cycle = np.append(arrived_vehicles_in_cycle, arrived_in_cycle)
        print(f'cycle {cycle}: {arrived_in_cycle} vehicles arrived, cycle real time: {(time.time() - start_cycle_time):.2f} seconds\t')
        arrived_in_cycle = 0
        cycle += 1
        start_cycle_time = time.time()
        
    
    step += 1

simulation_time = time.time() - start_time
print(f'simulation time: {simulation_time / 60} minutes')

# close sumo
traci.close()





# plots

# Create a single figure for all the plots
plt.figure(figsize=(12, 8))

# Create the first subplot for running vehicles and vehicles in the protected region
plt.subplot(2, 1, 1)  # 2 rows, 1 column, subplot 1
plt.plot(sim_time_step, running_vehicles, label="Running Vehicles", color='blue')
plt.plot(sim_time_step, vehicles_in_protected_region, label="Vehicles in Protected Region", color='red')
plt.title("Number of Running Vehicles and Vehicles in Protected Region")
plt.xlabel("Time")
plt.ylabel("Count")
plt.legend()

# Create the second subplot for arrived vehicles
plt.subplot(2, 1, 2)  # 2 rows, 1 column, subplot 2
plt.plot(sim_time_step, arrived_vehicles_in_cycle, label="Arrived Vehicles", color='blue')
plt.title("Arrived Vehicles")
plt.xlabel("Time")
plt.ylabel("Count")
plt.legend()

# Show the combined plot
plt.tight_layout()
plt.show()


array_of_matrices = np.array(matrix_list)
max = np.max(array_of_matrices)
heatmap.plot_dynamic_heatmap(array_of_matrices, vmin=0, vmax=1000, cmap="coolwarm")