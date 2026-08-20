import argparse
import os
import sys

from src.utils import load_config, setup_directories
from src.dataset_builder import DatasetBuilder

def main():
    parser = argparse.ArgumentParser(description="Extract ML-ready position dataset from combined PGN.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N games for testing")
    parser.add_argument("--batch-size", type=int, default=20000, help="Number of positions per chunk")
    args = parser.parse_args()

    base_path = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_path, "config.json"))
    
    username = config.get("username")
    output_directory = config.get("output_directory", "data")
    
    if not username:
        print("Error: 'username' not found in config.json")
        sys.exit(1)

    data_dir = os.path.join(base_path, output_directory)
    setup_directories(data_dir, username)
    
    combined_pgn = os.path.join(data_dir, "games", username, "combined", "all_games.pgn")
    positions_dir = os.path.join(data_dir, "positions")
    
    if not os.path.exists(combined_pgn):
        print(f"Error: Combined PGN not found at {combined_pgn}")
        print("Please run 'fetch_games.py' first.")
        sys.exit(1)
        
    print(f"Starting Position Dataset Builder for {username}...")
    if args.limit:
        print(f"LIMIT ACTIVE: Only processing first {args.limit} games.")
        
    builder = DatasetBuilder(
        username=username,
        combined_pgn_path=combined_pgn,
        output_dir=positions_dir,
        batch_size=args.batch_size
    )
    
    builder.build(max_games=args.limit)
    builder.print_report()

if __name__ == "__main__":
    main()
