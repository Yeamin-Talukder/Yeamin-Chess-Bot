import os
import glob
import pandas as pd
import io
import chess.pgn
from src.pgn_parser import parse_game_pgn
import logging

def process_raw_data(username: str, base_dir: str):
    """
    Reads all raw PGN files, combines them, and generates a structured CSV dataset.
    """
    logger = logging.getLogger('chess_downloader')
    raw_dir = os.path.join(base_dir, "games", username, "raw")
    combined_dir = os.path.join(base_dir, "games", username, "combined")
    processed_dir = os.path.join(base_dir, "processed")
    
    pgn_files = sorted(glob.glob(os.path.join(raw_dir, "*.pgn")))
    
    if not pgn_files:
        logger.warning(f"No raw PGN files found in {raw_dir}")
        return

    all_games_path = os.path.join(combined_dir, "all_games.pgn")
    csv_path = os.path.join(processed_dir, "games.csv")
    
    logger.info(f"Processing {len(pgn_files)} raw PGN files...")
    
    games_data = []
    
    # We will write to the combined PGN and process the CSV data simultaneously
    with open(all_games_path, 'w', encoding='utf-8') as combined_out:
        for pgn_file in pgn_files:
            with open(pgn_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if content.strip():
                    combined_out.write(content)
                    combined_out.write("\n\n")
                
                # Now parse the games from this file to populate the dataset
                f.seek(0)
                while True:
                    game = chess.pgn.read_game(f)
                    if game is None:
                        break
                    
                    # Convert game back to PGN string to pass to parser
                    # or just extract headers directly here to save time
                    headers = game.headers
                    
                    # Using chess.pgn.StringExporter to get the exact raw string representation
                    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
                    pgn_str = game.accept(exporter)
                    
                    game_dict = {
                        "game_id": headers.get("Link", "").split("/")[-1] if "Link" in headers else "",
                        "date": headers.get("UTCDate", headers.get("Date", "")),
                        "white": headers.get("White", ""),
                        "black": headers.get("Black", ""),
                        "white_rating": headers.get("WhiteElo", ""),
                        "black_rating": headers.get("BlackElo", ""),
                        "result": headers.get("Result", ""),
                        "time_control": headers.get("TimeControl", ""),
                        "rated": headers.get("Event", "").lower().find("rated") != -1 or "Rated" in headers.get("Event", ""),
                        "eco": headers.get("ECO", ""),
                        "opening": headers.get("ECOUrl", "").split("/")[-1].replace("-", " ") if "ECOUrl" in headers else "",
                        "termination": headers.get("Termination", ""),
                        "event": headers.get("Event", ""),
                        "pgn": pgn_str
                    }
                    games_data.append(game_dict)
                    
    logger.info(f"Extracted {len(games_data)} games.")
    
    # Build dataframe and save to CSV
    if games_data:
        df = pd.DataFrame(games_data)
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        logger.info(f"Saved structured dataset to {csv_path}")
    else:
        logger.warning("No games were parsed.")

