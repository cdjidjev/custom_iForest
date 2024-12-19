import pandas as pd
import numpy as np
import random
import os
import re
from collections import defaultdict

# Example function to generate the service_port_dict
def generate_service_port_dict(directory_path):
    service_port_dict = defaultdict(lambda: defaultdict(int))
    pattern = re.compile(r'(\d+)/([A-Z0-9]+)')

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as file:
                for line in file:
                    matches = pattern.findall(line)
                    for port, service in matches:
                        service_port_dict[service][port] += 1
    return service_port_dict

# Calculate distributions
def calculate_distributions(service_port_dict):
    total_services = sum(sum(ports.values()) for ports in service_port_dict.values())
    service_distribution = {service: sum(ports.values()) / total_services for service, ports in service_port_dict.items()}

    port_distributions = {}
    for service, ports in service_port_dict.items():
        total_ports = sum(ports.values())
        port_distributions[service] = {int(port): count / total_ports for port, count in ports.items()}
    
    return service_distribution, port_distributions

def generate_scan_data_1_old(num_ips, num_scans_per_ip, norm_list_length_distribution, service_distribution, port_distributions, anom_list_length_distribution, anom_service_distribution, anom_port_distribution, anomaly_rate=0.1, seed=None):
    
    np.random.seed(seed)
    random.seed(seed)

    def generate_ipv4():
        #random.seed(None)
        return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

    data = []
    global_scan_id = 0  # Initialize a global scan ID counter

    for ip_id in range(num_ips):  # `ip_id` will serve as the unique identifier for each IP
        ipv4 = generate_ipv4()
        # Determine if this IP will exhibit anomalous behavior
        is_anomalous_ip = np.random.rand() < anomaly_rate
        
        if is_anomalous_ip:
            list_length_distribution = anom_list_length_distribution
            service_dist = anom_service_distribution
            port_dist = anom_port_distribution
        else:
            list_length_distribution = norm_list_length_distribution
            service_dist = service_distribution
            port_dist = port_distributions
        
        # Pre-select services and ports for this IP
        list_length = np.random.choice(np.arange(1, len(list_length_distribution) + 1), p=list_length_distribution)
        services_chosen = np.random.choice(list(service_dist.keys()), list_length, p=list(service_dist.values()), replace=False)
        ports_chosen = {service: np.random.choice(list(port_dist[service].keys()), p=list(port_dist[service].values())) for service in services_chosen}

        for _ in range(num_scans_per_ip):
            if is_anomalous_ip:
                # Introduce anomalies by changing the behavior
                current_length = np.random.choice(np.arange(1, len(anom_list_length_distribution) + 1), p=anom_list_length_distribution)
                current_services = np.random.choice(list(anom_service_distribution.keys()), current_length, p=list(anom_service_distribution.values()), replace=False)
                current_ports = [np.random.choice(list(anom_port_distribution[service].keys()), p=list(anom_port_distribution[service].values())) for service in current_services]
            else:
                current_length = random.randint(1, list_length)
                current_services = random.sample(services_chosen.tolist(), current_length)
                current_ports = [ports_chosen[service] for service in current_services]

            anomaly = 1 if is_anomalous_ip else 0
            # Append the `ip_id` as the unique identifier for this IP
            data.append([ipv4, ip_id, global_scan_id, current_ports, current_services, anomaly])
            global_scan_id += 1  # Increment the global scan ID

    # Add 'ip_id' to columns to store the unique identifier per IP address
    return pd.DataFrame(data, columns=['ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'])

def generate_scan_data_1(num_ips, num_scans_per_ip, norm_list_length_distribution, common_service_distribution, common_port_distribution, anom_list_length_distribution, anomaly_service_distribution, anomaly_port_distribution, anomaly_rate=0.1, seed=None):
    np.random.seed(seed)
    random.seed(seed)

    def generate_ipv4():
        return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

    data = []
    global_scan_id = 0  # Initialize a global scan ID counter

    num_anomolous_ips = max(1, int(num_ips * anomaly_rate))
    anom_ip_id = random.sample(range(num_ips), num_anomolous_ips)

    for ip_id in range(num_ips):  # `ip_id` will serve as the unique identifier for each IP
        ipv4 = generate_ipv4()
        # Determine if this IP will exhibit anomalous behavior
        is_anomalous_ip = (ip_id in anom_ip_id)
        
        # Define the distributions for normal and anomalous behaviors
        if is_anomalous_ip:
            # Use normal activity for most scans but introduce anomalies
            normal_scans = int(num_scans_per_ip * 0.8)  # 80% normal scans
            anomalous_scans = num_scans_per_ip - normal_scans
        else:
            normal_scans = num_scans_per_ip
            anomalous_scans = 0

        # Generate normal activity scans
        for _ in range(normal_scans):
            list_length = np.random.choice(np.arange(1, len(norm_list_length_distribution) + 1), p=norm_list_length_distribution)
            list_length = min(list_length, len(common_service_distribution))  # Adjust length
            services_chosen = np.random.choice(list(common_service_distribution.keys()), list_length, p=list(common_service_distribution.values()), replace=False)
            ports_chosen = {service: np.random.choice(list(common_port_distribution[service].keys()), p=list(common_port_distribution[service].values())) for service in services_chosen}
            data.append([ipv4, ip_id, global_scan_id, list(ports_chosen.values()), list(services_chosen), 0])  # Normal scan, anomaly=0
            global_scan_id += 1

        # Generate anomalous activity scans (introduce uncommon services/ports)
        for _ in range(anomalous_scans):
            list_length = np.random.choice(np.arange(1, len(anom_list_length_distribution) + 1), p=anom_list_length_distribution)
            list_length = min(list_length, len(anomaly_service_distribution))  # Adjust length
            services_chosen = np.random.choice(list(anomaly_service_distribution.keys()), list_length, p=list(anomaly_service_distribution.values()), replace=False)
            ports_chosen = {service: np.random.choice(list(anomaly_port_distribution[service].keys()), p=list(anomaly_port_distribution[service].values())) for service in services_chosen}
            data.append([ipv4, ip_id, global_scan_id, list(ports_chosen.values()), list(services_chosen), 1])  # Anomalous scan, anomaly=1
            global_scan_id += 1

    # Return the data as a DataFrame
    return pd.DataFrame(data, columns=['ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'])

def generate_scan_data_2(num_ips, num_scans_per_ip, norm_list_length_distribution, common_service_distribution, common_port_distribution, anom_list_length_distribution, anomaly_service_distribution, anomaly_port_distribution, anomaly_rate=0.1, seed=None):
    np.random.seed(seed)
    random.seed(seed)

    def generate_ipv4():
        return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

    # Create typical_service_ports from the common_port_distribution
    typical_service_ports = {}
    for service in common_port_distribution:
        # Get the most common ports for each service from the distribution
        ports = list(common_port_distribution[service].keys())
        probabilities = list(common_port_distribution[service].values())
        # Select ports with probability > threshold (e.g., 0.1) as typical
        typical_ports = [port for port, prob in zip(ports, probabilities) if prob > 0.1]
        if not typical_ports:  # If no ports meet threshold, take the top 2 most likely ports
            typical_ports = [ports[i] for i in np.argsort(probabilities)[-2:]]
        typical_service_ports[service] = typical_ports

    data = []
    global_scan_id = 0
    num_anomolous_ips = max(1, int(num_ips * anomaly_rate))
    anom_ip_id = random.sample(range(num_ips), num_anomolous_ips)

    for ip_id in range(num_ips):
        ipv4 = generate_ipv4()
        is_anomalous_ip = (ip_id in anom_ip_id)
        
        # Establish normal behavior pattern for this IP
        base_services = np.random.choice(
            list(common_service_distribution.keys()),
            size=np.random.randint(1, 4),
            p=list(common_service_distribution.values()),
            replace=False
        )
        
        # For each scan from this IP
        for _ in range(num_scans_per_ip):
            # Determine if this particular scan will be anomalous
            is_anomalous_scan = is_anomalous_ip and np.random.rand() < 0.2  # 20% of scans from anomalous IPs
            
            if is_anomalous_scan:
                # Generate anomalous behavior by using atypical ports for services
                list_length = np.random.choice(
                    np.arange(1, len(anom_list_length_distribution) + 1),
                    p=anom_list_length_distribution
                )
                
                # Select services from base services to make anomalous
                current_services = np.random.choice(base_services, size=min(list_length, len(base_services)), replace=False)
                current_ports = []
                
                for service in current_services:
                    # Get typical ports for this service
                    typical_ports = typical_service_ports[service]
                    # Get all possible ports from the port distribution
                    all_ports = list(common_port_distribution[service].keys())
                    # Generate an atypical port (one that's not in typical_ports)
                    available_ports = [p for p in all_ports if p not in typical_ports]
                    if not available_ports:  # If no atypical ports available, generate a random high port
                        atypical_port = random.randint(10000, 65535)
                    else:
                        atypical_port = np.random.choice(available_ports)
                    current_ports.append(atypical_port)
                
                anomaly = 1
                
            else:
                # Generate normal behavior
                list_length = np.random.choice(
                    np.arange(1, len(norm_list_length_distribution) + 1),
                    p=norm_list_length_distribution
                )
                
                # Use subset of base services
                current_services = np.random.choice(
                    base_services,
                    size=min(list_length, len(base_services)),
                    replace=False
                )
                
                # Use typical ports for these services
                current_ports = []
                for service in current_services:
                    port = np.random.choice(typical_service_ports[service])
                    current_ports.append(port)
                
                anomaly = 0

            data.append([ipv4, ip_id, global_scan_id, current_ports, current_services.tolist(), anomaly])
            global_scan_id += 1

    return pd.DataFrame(data, columns=['ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'])
def extract_typical_ports(service_port_dict):
    """
    Extracts the most common port for each service from a nested defaultdict structure.
    """
    typical_ports = {}
    for service, ports in service_port_dict.items():
        # Find the port with the highest frequency
        typical_port = max(ports.items(), key=lambda x: x[1])[0]
        typical_ports[service] = int(typical_port)  # Ensure the port is an integer
    return typical_ports

def generate_scan_data_3(num_ips, num_scans_per_ip, typical_service_port_map, anomaly_rate=0.1, anomalous_scan_prob=0.2, seed=None):
    """
    Generates synthetic scan data with specific anomalies: services running on atypical ports.

    Parameters:
    - num_ips (int): Number of IPs to generate.
    - num_scans_per_ip (int): Number of scans per IP.
    - typical_service_port_map (dict): Mapping of each service to its typical port.
    - anomaly_rate (float): Proportion of IPs to generate with anomalies.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Data with columns 'ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'.
    """
    #print('generate_scan_data_3().')
    np.random.seed(seed)
    random.seed(seed)
    
    def generate_ipv4():
        return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))
    
    data = []
    global_scan_id = 0
    services = list(typical_service_port_map.keys())

    # choose ips to become df[df['anomaly']==1].head()
    num_anomolous_ips = max(1, int(num_ips * anomaly_rate))
    anom_ip_id = random.sample(range(num_ips), num_anomolous_ips)

    for ip_id in range(num_ips):
        ipv4 = generate_ipv4()
        is_anomalous_ip = (ip_id in anom_ip_id) # Determine if this IP is anomalous
        '''if ip_id == 0:
            is_anomalous_ip = True'''
        #print('is_anomalous_ip', is_anomalous_ip)
        #if is_anomalous_ip:
            #print('Anomalous ip:', ipv4)

        # For each IP, create a base set of typical services and ports
        #base_services = random.sample(services, k=random.randint(3, len(services)))
        #base_ports = [typical_service_port_map[service] for service in base_services]

        scan_services = services.copy()
        scan_ports = [typical_service_port_map[service] for service in scan_services]

        for scan_id in range(num_scans_per_ip):
            scan_ports_modified = scan_ports.copy()
            if is_anomalous_ip and scan_id >= int(num_scans_per_ip * (1-anomalous_scan_prob)):  # anomalous_scan_prob anomalous scans for anomalous IPs
                # Choose two unique random indices to swap ports
                idx1, idx2 = random.sample(range(len(scan_ports)), 2)
                # Swap the ports at the chosen indices
                scan_ports_modified[idx1], scan_ports_modified[idx2] = scan_ports_modified[idx2], scan_ports_modified[idx1]
                anomaly_flag = 1  # Label this scan as anomalous
            else:
                anomaly_flag = 0

            # Append each scan for this IP
            data.append([ipv4, ip_id, global_scan_id, scan_ports_modified, scan_services, anomaly_flag])
            global_scan_id += 1

    # Convert to DataFrame
    return pd.DataFrame(data, columns=['ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'])

def generate_scan_data_mismatch(num_ips, num_scans_per_ip, norm_list_length_distribution, common_service_distribution, common_port_distribution, anom_list_length_distribution, anomaly_service_distribution, anomaly_port_distribution, anomaly_rate=0.1, seed=None):
    """
    Generates synthetic scan data with anomalies as services running on atypical ports.

    Parameters:
    - num_ips (int): Number of IPs to generate.
    - num_scans_per_ip (int): Number of scans per IP.
    - norm_list_length_distribution (list): Probability distribution for the number of services in normal scans.
    - common_service_distribution (dict): Distribution of common services.
    - common_port_distribution (dict): Port distributions for each common service.
    - anom_list_length_distribution (list): Probability distribution for the number of services in anomalous scans.
    - anomaly_service_distribution (dict): Distribution of services for anomalous scans.
    - anomaly_port_distribution (dict): Port distributions for each anomalous service.
    - anomaly_rate (float): Proportion of IPs to be generated with anomalies.
    - seed (int, optional): Random seed for reproducibility.

    Returns:
    - pd.DataFrame: Data with columns 'ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'.
    """
    np.random.seed(seed)
    random.seed(seed)

    def generate_ipv4():
        return ".".join(map(str, (random.randint(0, 255) for _ in range(4))))

    data = []
    global_scan_id = 0

    for ip_id in range(num_ips):
        ipv4 = generate_ipv4()
        is_anomalous_ip = np.random.rand() < anomaly_rate
        anomaly_flag = 1 if is_anomalous_ip else 0
        
        # Define number of normal and anomalous scans
        if is_anomalous_ip:
            normal_scans = int(num_scans_per_ip * 0.8)  # 80% normal scans for anomalous IPs
            anomalous_scans = num_scans_per_ip - normal_scans
        else:
            normal_scans = num_scans_per_ip
            anomalous_scans = 0

        # Generate normal scans
        for _ in range(normal_scans):
            list_length = np.random.choice(np.arange(1, len(norm_list_length_distribution) + 1), p=norm_list_length_distribution)
            list_length = min(list_length, len(common_service_distribution))  # Limit length to available services
            services_chosen = np.random.choice(list(common_service_distribution.keys()), list_length, p=list(common_service_distribution.values()), replace=False)
            ports_chosen = [np.random.choice(list(common_port_distribution[service].keys()), p=list(common_port_distribution[service].values())) for service in services_chosen]
            data.append([ipv4, ip_id, global_scan_id, ports_chosen, list(services_chosen), 0])
            global_scan_id += 1

        # Generate anomalous scans with atypical service-port pairs
        for _ in range(anomalous_scans):
            list_length = np.random.choice(np.arange(1, len(anom_list_length_distribution) + 1), p=anom_list_length_distribution)
            list_length = min(list_length, len(anomaly_service_distribution))  # Adjust length to available services
            services_chosen = np.random.choice(list(anomaly_service_distribution.keys()), list_length, p=list(anomaly_service_distribution.values()), replace=False)
            ports_chosen = [np.random.choice(list(anomaly_port_distribution[service].keys()), p=list(anomaly_port_distribution[service].values())) for service in services_chosen]
            data.append([ipv4, ip_id, global_scan_id, ports_chosen, list(services_chosen), 1])
            global_scan_id += 1

    # Return the generated data as a DataFrame
    return pd.DataFrame(data, columns=['ipv4', 'ip_id', 'scan_id', 'port', 'service', 'anomaly'])

def flatten_columns(df, column):
    df_flattened = df.copy()
    
    # Initialize a dictionary to hold the new columns
    new_columns = {}

    # Accumulate the columns to add in the dictionary
    for i in range(len(df)):
        for item in df.at[i, column]:
            col_name = f"{column}_{item}"
            if col_name not in new_columns:
                new_columns[col_name] = [0] * len(df)  # Initialize the column with 0s
            new_columns[col_name][i] = 1  # Set to 1 where appropriate

    # Convert the dictionary to a DataFrame
    new_columns_df = pd.DataFrame(new_columns)

    # Concatenate with the original DataFrame, excluding the original column
    df_flattened = pd.concat([df_flattened.drop(columns=[column]), new_columns_df], axis=1)
    
    return df_flattened


def preprocess_customiForest(df):
    # Drop the 'anomaly' column and prepare the data for Isolation Forest
    if 'scan_id' in df.columns:
        features = df.drop(columns=['scan_id'])
    else:
        features = df.copy()  # If there's no 'scan_id' column, just copy the DataFrame


    # Split IPv4 into four octets
    features[['octet1', 'octet2', 'octet3', 'octet4']] = df['ipv4'].str.split('.', expand=True)
    features[['octet1', 'octet2', 'octet3', 'octet4']] = features[['octet1', 'octet2', 'octet3', 'octet4']].astype(int)

    # Drop the original 'ipv4' column
    features = features.drop(columns=['ipv4', 'anomaly'])

    # Flatten the 'port' and 'service' columns
    features_flat = flatten_columns(features, 'port')
    features_flat = flatten_columns(features_flat, 'service')

    return features_flat

from collections import Counter

def aggregate_features(group):
    # Extract all unique service names from the 'service_name' column
    unique_services = sorted(set(service for sublist in group['service'] for service in sublist))

    # Create a mapping from service name to number
    service_to_number = {service: idx + 1 for idx, service in enumerate(unique_services)}

    # Function to map service names to numbers
    map_services_to_numbers = lambda services_list: [service_to_number[service] for service in services_list]

    # Apply the mapping function to the 'service_name' column
    group['service_number'] = group['service'].apply(map_services_to_numbers)
    ports = [port for sublist in group['port'] for port in sublist]
    service_nums = [service for sublist in group['service_number'] for service in sublist]
    
    # Count unique counts per IP address
    unique_port_count_per_id = group['port'].apply(lambda x: len(set(x)))
    unique_serv_count_per_id = group['service_number'].apply(lambda x: len(set(x)))
    
    # Calculate port and service frequencypreprocess_customiForest
    port_frequency = Counter(ports)
    service_frequency = Counter(service_nums)
    
    # Calculate port ranges
    well_known_ports = sum(1 for port in ports if 0 <= port <= 1023)
    registered_ports = sum(1 for port in ports if 1024 <= port <= 49151)
    dynamic_private_ports = sum(1 for port in ports if 49152 <= port <= 65535)
    
    # Calculate port and service variance
    port_var = pd.Series(ports).var()
    serv_var = pd.Series(service_nums).var()

    return pd.Series({
        'most_frequent_port': port_frequency.most_common(1)[0][0] if ports else None,
        'most_frequent_service': service_frequency.most_common(1)[0][0] if service_nums else None,
        'unique_port_count_per_id': unique_port_count_per_id.mean(),
        'unique_service_count_per_id': unique_serv_count_per_id.mean(),
        'well_known_ports': well_known_ports,
        'registered_ports': registered_ports,
        'dynamic_private_ports': dynamic_private_ports,
        'port_var': port_var,
        'serv_var': serv_var
    })

def split_ipv4_into_octets(df):
    """
    Splits the IPv4 column into four octet columns and drops the original 'ipv4' column.
    
    Parameters:
    - df: DataFrame containing an 'ipv4' column
    
    Returns:
    - DataFrame with 'octet1', 'octet2', 'octet3', and 'octet4' columns and the 'ipv4' column removed.
    """
    # Split IPv4 into four octets
    df[['octet1', 'octet2', 'octet3', 'octet4']] = df['ipv4'].str.split('.', expand=True)
    
    # Convert octets to float
    df[['octet1', 'octet2', 'octet3', 'octet4']] = df[['octet1', 'octet2', 'octet3', 'octet4']].astype(float)
    
    # Drop the original 'ipv4' column
    df = df.drop(columns=['ipv4'])
    
    return df

def summarize_features(df):
    ## Get all unique ports and services across the dataset
    all_ports = sorted({port for ports in df['port'] for port in ports})
    all_services = sorted({service for services in df['service'] for service in services})
    
    # Initialize lists to store the results
    results = []
    
    # Iterate through each IP address
    for ip, group in df.groupby("ip_id"):
        # Flatten the list of ports and services for this IP
        ip_ports = [port for ports in group['port'] for port in ports]
        ip_services = [service for services in group['service'] for service in services]
        
        # Count occurrences of each port and service
        port_counts = Counter(ip_ports)
        service_counts = Counter(ip_services)
        
        # Prepare the result row with counts for each unique port and service
        result = {"ip_id": ip}
        
        # Populate counts for each port, defaulting to 0 if the port is absent
        for port in all_ports:
            result[f"port_{port}"] = int(port_counts.get(port, 0))
        # Populate counts for each service, defaulting to 0 if the service is absent
        for service in all_services:
            result[f"service_{service}"] = int(service_counts.get(service, 0))
        
        results.append(result)
    
    # Convert results to a DataFrame
    return pd.DataFrame(results)       

def pairs(df):
    # Flatten the services into unique integer identifiers
    unique_services = pd.Series([service for sublist in df['service'] for service in sublist]).unique()
    service_to_id = {service: idx for idx, service in enumerate(unique_services)}

    # Function to transform each row
    transformed_data = []
    for index, row in df.iterrows():
        ip_id = row['ip_id']  # Use 'ip_id' instead of 'ipv4'
        ports = row['port']  # List of ports
        services = row['service']  # List of services
        
        # Ensure ports and services are paired correctly
        if len(ports) != len(services):
            raise ValueError(f"Mismatched lengths for ports and services in row {index}")

        for port, service in zip(ports, services):
            transformed_row = {
                'ip_id': ip_id,  # Include 'ip_id' in each transformed row
                'port': port,
                'service_id': service_to_id[service]  # Map service to unique ID
            }
            transformed_data.append(transformed_row)

    # Convert transformed data to a DataFrame
    transformed_df = pd.DataFrame(transformed_data)

    return transformed_df