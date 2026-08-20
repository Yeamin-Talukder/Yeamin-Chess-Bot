import os
import json
import logging

def setup_directories(base_dir: str, username: str):
    """
    Creates necessary directories for storing games and processed data.
    """
    dirs = [
        os.path.join(base_dir, "games", username, "raw"),
        os.path.join(base_dir, "games", username, "combined"),
        os.path.join(base_dir, "processed"),
        os.path.join(base_dir, "positions"),
        os.path.join(base_dir, "analysis")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    
    # Create logs directory
    os.makedirs(os.path.join(os.path.dirname(base_dir), "logs"), exist_ok=True)

def load_config(config_path: str = "config.json") -> dict:
    """
    Loads configuration from a JSON file.
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_path}' not found.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_path}' is not valid JSON.")
        return {}

def setup_logger(log_file: str):
    """
    Sets up the logging configuration.
    """
    logger = logging.getLogger('chess_downloader')
    logger.setLevel(logging.INFO)
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
