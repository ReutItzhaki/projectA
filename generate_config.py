import os
import subprocess
from datetime import datetime

# Paths to required files
network_file = "SUMO_files/network.net.xml"
traffic_light_file = "SUMO_files/traffic_lights.tll.xml"
route_script = "Python_files/generate_routes_xml.py"
duarouter_executable = "duarouter"  # Assuming duarouter is available in the system path

# Directory to save output files
output_dir = "sumo_files"
os.makedirs(output_dir, exist_ok=True)

# Current timestamp for the generated comment
timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Loop to generate files
for i in range(1, 301):
    # Define the route file path
    route_file = os.path.join(output_dir, f"routes{i}.rou.xml")
    
    # Generate the route file using the script (without the --ignore-errors flag)
    try:
        with open(route_file, "w") as rf:
            subprocess.run(
                ["python", route_script, "--output", route_file],
                stdout=rf,  # Capture stdout to the route file
                stderr=subprocess.PIPE,  # Capture stderr to help with debugging
                check=True
            )
            print(f"Generated route file: {route_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating route file: {route_file}")
        print("Standard Output:", e.stdout.decode() if e.stdout else "None")
        print("Standard Error:", e.stderr.decode() if e.stderr else "None")
        continue  # Skip to the next iteration on error
    
    # Run duarouter to validate and correct the routes (with --ignore-errors)
    corrected_route_file = os.path.join(output_dir, f"routes{i}_corrected.rou.xml")
    subprocess.run(
        [duarouter_executable, "-n", network_file, "-r", route_file, "--ignore-errors", "-o", corrected_route_file],
        check=True
    )
    
    # Create the SUMO configuration file
    sumo_config_file = os.path.join(output_dir, f"sumoconfig{i}.sumocfg")
    with open(sumo_config_file, "w") as config:
        config.write(f"""<?xml version="1.0" encoding="UTF-8"?>

<!-- generated on {timestamp} by Python script -->

<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">

    <input>
        <net-file value="../{network_file}"/>
        <route-files value="../{corrected_route_file}"/>
    </input>

</configuration>
""")
        #<additional-files value="{traffic_light_file}"/>
