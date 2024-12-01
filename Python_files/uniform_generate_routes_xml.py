#!/usr/bin/env python3
from itertools import product
import random

# Define simulation parameters
ind = 0
num_trips_per_pair = 3  # Fixed number of trips per start-destination pair
simulation_time = 4000

# Define starts (city perimeter) and destinations
starts = [f'f{i}' for i in range(1, 25)]  # Perimeter points
dests = [f'd{i}' for i in range(1, 101)]  # Destination points

# Generate all paths (start, destination pairs)
paths = list(product(starts, dests))

# Print SUMO XML format
print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
print('\t<vType id="car" lcStrategic="120" lcCooperative="1.0"/>')

# Uniformly generate trips
for start, dest in paths:
    for _ in range(num_trips_per_pair):  # Create the same number of trips for each pair
        depart_time = random.uniform(0, simulation_time)  # Uniformly distributed departure time
        print(f'\t<trip id="{ind}" depart="{depart_time}" from="{start}" to="{dest}" type="car"/>')
        ind += 1

print('</routes>')