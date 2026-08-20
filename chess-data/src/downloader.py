import os
import logging
from typing import Tuple
from src.chess_api import ChessAPI

class ChessDownloader:
    def __init__(self, username: str, base_dir: str):
        self.username = username
        self.base_dir = base_dir
        self.api = ChessAPI(username)
        self.logger = logging.getLogger('chess_downloader')
        self.raw_dir = os.path.join(base_dir, "games", username, "raw")

    def download_archives(self, force: bool = False) -> Tuple[int, int, int]:
        """
        Downloads the monthly archives.
        Returns a tuple: (total_games, new_games, skipped_games)
        """
        self.logger.info("Checking available archives...")
        print("Checking available archives...\n")
        
        try:
            archives = self.api.get_available_archives()
        except Exception as e:
            self.logger.error(f"Failed to get archives: {e}")
            print(f"Failed to get archives: {e}")
            return 0, 0, 0
            
        total_games = 0
        new_games = 0
        skipped_existing = 0
        
        for url in archives:
            # URL format: https://api.chess.com/pub/player/{username}/games/{YYYY}/{MM}
            parts = url.split('/')
            year, month = parts[-2], parts[-1]
            month_label = f"{year}-{month}"
            
            file_path = os.path.join(self.raw_dir, f"{month_label}.pgn")
            
            if os.path.exists(file_path) and not force:
                # To get the count of skipped games, we could parse the file or just count the `[Event ` headers
                # but parsing every existing file might be slow. We'll just do a quick count.
                with open(file_path, 'r', encoding='utf-8') as f:
                    count = f.read().count('[Event ')
                skipped_existing += count
                total_games += count
                print(f"[OK] {month_label} - {count} games (skipped)")
                self.logger.info(f"Skipped {month_label} (already exists)")
                continue
                
            print(f"[DL] {month_label} - downloading...")
            self.logger.info(f"Downloading {month_label}...")
            
            data = self.api.fetch_monthly_archive(url)
            if not data or "games" not in data:
                print(f"[!] {month_label} - failed or empty")
                continue
                
            games = data["games"]
            num_games = len(games)
            
            # Combine PGNs
            pgn_content = ""
            for game in games:
                if "pgn" in game:
                    pgn_content += game["pgn"] + "\n\n"
                    
            if pgn_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(pgn_content)
                
            new_games += num_games
            total_games += num_games
            print(f"[OK] {month_label} - {num_games} games")
            
        return total_games, new_games, skipped_existing
