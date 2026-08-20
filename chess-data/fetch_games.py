import argparse
import os
import sys

from src.utils import load_config, setup_directories, setup_logger
from src.downloader import ChessDownloader
from src.processor import process_raw_data

def main():
    parser = argparse.ArgumentParser(description="Download and process Chess.com game archives.")
    parser.add_argument("--force", action="store_true", help="Force redownload of all archives")
    args = parser.parse_args()

    # Base path logic (assuming the script is run from the project root)
    base_path = os.path.dirname(os.path.abspath(__file__))
    config = load_config(os.path.join(base_path, "config.json"))
    
    username = config.get("username")
    output_directory = config.get("output_directory", "data")
    
    if not username:
        print("Error: 'username' not found in config.json")
        sys.exit(1)

    print(f"Chess.com Game Downloader\n" + "-" * 26 + "\n")
    print(f"Username: {username}\n")
    
    # Resolve the data directory
    data_dir = os.path.join(base_path, output_directory)
    
    setup_directories(data_dir, username)
    logger = setup_logger(os.path.join(base_path, "logs", "downloader.log"))
    
    logger.info(f"Starting downloader for {username}")
    
    # Download
    downloader = ChessDownloader(username, data_dir)
    total, new, skipped = downloader.download_archives(force=args.force)
    
    print("\n" + "-" * 26)
    print(f"Total games: {total}")
    print(f"New games: {new}")
    print(f"Skipped existing: {skipped}")
    print("-" * 26 + "\n")
    
    # Process
    if new > 0 or args.force or not os.path.exists(os.path.join(data_dir, "processed", "games.csv")):
        print("Processing dataset...\n")
        process_raw_data(username, data_dir)
    
    print(f"Dataset saved to:\n{os.path.join(data_dir, 'games', username)}\n")

if __name__ == "__main__":
    main()
