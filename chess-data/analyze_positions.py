import os
import json
import argparse
import multiprocessing
from src.analysis_builder import AnalysisBuilder

def load_config(config_path="config.json"):
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description="Analyze chess positions with Stockfish (Distributed Mode).")
    parser.add_argument("--data-dir", default="data", help="Directory where data is stored")
    parser.add_argument("--config", default="config.json", help="Path to configuration file")
    parser.add_argument("--forward", action="store_true", help="Process forwards (0 -> end) instead of backwards")
    parser.add_argument("--workers", type=int, help="Override number of multiprocessing workers")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.workers:
        config["num_workers"] = args.workers

    print("Initializing Stockfish Analysis Pipeline...")
    builder = AnalysisBuilder(data_dir=args.data_dir, config=config)
    
    # Run in reverse by default (to complement Colab)
    run_reverse = not args.forward
    builder.build(reverse=run_reverse)

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
