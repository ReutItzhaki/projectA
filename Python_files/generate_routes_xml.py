#!/usr/bin python3
from itertools import product
import random

car_number = 2
ind = 0
times = [i*8100 for i in range(2)]
starts = [f'f{i}' for i in range(1, 25)]
dests = [f'd{i}' for i in range(1, 101)]
mu = 2000
sigma = 500
simulation_time = 4000
paths = list(product(starts, dests))
all_dests = list(product(dests, dests))
print('<routes xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/routes_file.xsd">')
print(f'\t<vType id="car" lcStrategic="120" lcCooperative="1.0"/>')
#for i in range(len(times) - 1):
for (start, dest) in paths:
    print(f'\t<trip id="{ind}" depart="{max(0, min(simulation_time, random.gauss(mu, sigma)))}" from="{start}" to="{dest}" type="car"/>')
    ind += 1
    print(f'\t<trip id="{ind}" depart="{max(0, min(simulation_time, random.gauss(mu, sigma)))}" from="{start}" to="{dest}" type="car"/>')
    ind += 1
    
    if(random.random() > 0.5):
        print(f'\t<trip id="{ind}" depart="{max(0, min(simulation_time, random.gauss(mu, sigma)))}" from="{start}" to="{dest}" type="car"/>')
        ind += 1
        
    if(random.random() > 0.5):  
        print(f'\t<trip id="{ind}" depart="{max(0, min(simulation_time, random.gauss(mu, sigma)))}" from="{start}" to="{dest}" type="car"/>')
        ind += 1  
                    
for dest1, dest2 in all_dests:
    if dest1 == dest2:
        continue
    
    print(f'\t<trip id="{ind}" depart="{max(0, min(simulation_time, random.gauss(mu, sigma)))}" from="-{dest1}" to="{dest2}" type="car"/>')
    ind += 1
print('</routes>') 

