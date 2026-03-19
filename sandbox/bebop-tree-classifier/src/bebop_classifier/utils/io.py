def read_config(file_path):
    # Function to read configuration from a YAML file
    import yaml
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_results(results, file_path):
    # Function to save results to a file
    import json
    with open(file_path, 'w') as file:
        json.dump(results, file)