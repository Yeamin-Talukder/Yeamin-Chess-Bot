import os
import chess.pgn
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.position_extractor import extract_positions_from_game

class DatasetBuilder:
    def __init__(self, username: str, combined_pgn_path: str, output_dir: str, batch_size: int = 20000):
        self.username = username
        self.combined_pgn_path = combined_pgn_path
        self.output_dir = output_dir
        self.batch_size = batch_size
        
        self.stats = {
            "Total games": 0,
            "Games skipped": 0,
            "Games played as White": 0,
            "Games played as Black": 0,
            "Total positions": 0,
            "White positions": 0,
            "Black positions": 0,
            "Rated games": 0,
            "Casual games": 0,
            "Bullet": 0,
            "Blitz": 0,
            "Rapid": 0,
            "Daily": 0,
            "Other": 0,
            "Invalid PGNs": 0,
            "Parsing errors": 0
        }

    def _update_game_stats(self, headers, is_white):
        self.stats["Total games"] += 1
        
        if is_white:
            self.stats["Games played as White"] += 1
        else:
            self.stats["Games played as Black"] += 1
            
        event = headers.get("Event", "").lower()
        if "rated" in event or headers.get("Event", "") == "Live Chess":
            self.stats["Rated games"] += 1
        else:
            self.stats["Casual games"] += 1
            
        time_class = headers.get("TimeControl", "")
        if "1/259200" in time_class or "/" in time_class or time_class == "1/86400": # very rough heuristic for daily
             # We rely on "Event" mostly for daily, but let's check TimeControl
             pass 
             
        # More robust variant: use "Event" string which usually has "Blitz", "Rapid", "Bullet", "Daily"
        if "bullet" in event:
            self.stats["Bullet"] += 1
        elif "blitz" in event:
            self.stats["Blitz"] += 1
        elif "rapid" in event:
            self.stats["Rapid"] += 1
        elif "daily" in event:
            self.stats["Daily"] += 1
        else:
            self.stats["Other"] += 1

    def build(self, max_games: int = None):
        csv_path = os.path.join(self.output_dir, "positions.csv")
        parquet_path = os.path.join(self.output_dir, "positions.parquet")
        
        # Clean previous files if they exist
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(parquet_path):
            os.remove(parquet_path)
            
        pos_id = 1
        current_batch = []
        pq_writer = None
        
        with open(self.combined_pgn_path, 'r', encoding='utf-8') as pgn_file:
            while True:
                if max_games and self.stats["Total games"] >= max_games:
                    break
                    
                try:
                    game = chess.pgn.read_game(pgn_file)
                except Exception as e:
                    self.stats["Invalid PGNs"] += 1
                    self.stats["Games skipped"] += 1
                    continue
                    
                if game is None:
                    break
                
                headers = game.headers
                white = headers.get("White", "")
                black = headers.get("Black", "")
                
                is_white = self.username.lower() == white.lower()
                is_black = self.username.lower() == black.lower()
                
                if not is_white and not is_black:
                    self.stats["Games skipped"] += 1
                    continue
                    
                self._update_game_stats(headers, is_white)
                
                # Extract
                positions, pos_id, errors = extract_positions_from_game(game, self.username, pos_id)
                self.stats["Parsing errors"] += errors
                
                current_batch.extend(positions)
                
                if len(current_batch) >= self.batch_size:
                    pq_writer = self._flush_batch(current_batch, csv_path, parquet_path, pq_writer)
                    current_batch = []
                    
            # flush remaining
            if current_batch:
                pq_writer = self._flush_batch(current_batch, csv_path, parquet_path, pq_writer)
                
        if pq_writer:
            pq_writer.close()
            
    def _flush_batch(self, batch, csv_path, parquet_path, pq_writer):
        df = pd.DataFrame(batch)
        
        # Update pos stats
        self.stats["Total positions"] += len(df)
        white_pos = len(df[df["your_color"] == "White"])
        black_pos = len(df[df["your_color"] == "Black"])
        
        self.stats["White positions"] += white_pos
        self.stats["Black positions"] += black_pos
        
        # Write CSV
        header = not os.path.exists(csv_path)
        df.to_csv(csv_path, mode='a', header=header, index=False, encoding='utf-8-sig')
        
        # Write Parquet
        table = pa.Table.from_pandas(df)
        if pq_writer is None:
            pq_writer = pq.ParquetWriter(parquet_path, table.schema)
        pq_writer.write_table(table)
        
        print(f"Flushed batch of {len(batch)} positions. Total positions so far: {self.stats['Total positions']}")
        return pq_writer
        
    def print_report(self):
        avg_pos = 0
        if self.stats["Total games"] > 0:
            avg_pos = self.stats["Total positions"] / self.stats["Total games"]
            
        report = f"""
========================================
POSITION DATASET COMPLETE
========================================

Games processed:       {self.stats['Total games']}
Games skipped:         {self.stats['Games skipped']}

Positions extracted:   {self.stats['Total positions']}

White positions:       {self.stats['White positions']}
Black positions:       {self.stats['Black positions']}

Average positions/game: {avg_pos:.1f}

Rated games:           {self.stats['Rated games']}
Casual games:          {self.stats['Casual games']}

Bullet:                {self.stats['Bullet']}
Blitz:                 {self.stats['Blitz']}
Rapid:                 {self.stats['Rapid']}
Daily:                 {self.stats['Daily']}
Other:                 {self.stats['Other']}

Invalid PGNs:          {self.stats['Invalid PGNs']}
Parsing errors:        {self.stats['Parsing errors']}

Dataset:
{os.path.join(self.output_dir, 'positions.parquet')}

Next stage:
Stockfish position analysis
========================================
"""
        print(report)
