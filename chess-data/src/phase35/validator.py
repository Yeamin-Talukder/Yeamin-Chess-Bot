import pandas as pd
import json
import os

def validate_data(df: pd.DataFrame, output_dir: str):
    """
    Validates the dataset and outputs a JSON summary of data quality.
    """
    print("Validating dataset...")
    
    total_positions = len(df)
    unique_positions = df['position_id'].nunique() if 'position_id' in df.columns else total_positions
    duplicate_positions = total_positions - unique_positions
    
    unique_games = df['game_id'].nunique() if 'game_id' in df.columns else 0
    
    missing_fens = int(df['fen'].isnull().sum()) if 'fen' in df.columns else 0
    invalid_fens = int(df[~df['fen'].str.contains(' ', na=False)].shape[0]) if 'fen' in df.columns else 0
    
    missing_moves = int(df['your_move'].isnull().sum()) if 'your_move' in df.columns else 0
    missing_color = int(df['your_color'].isnull().sum()) if 'your_color' in df.columns else 0
    
    missing_engine_eval = int(df['stockfish_best_eval_cp'].isnull().sum() & df['stockfish_best_eval_mate'].isnull().sum()) if 'stockfish_best_eval_cp' in df.columns else 0
    missing_cpl = int(df['centipawn_loss'].isnull().sum()) if 'centipawn_loss' in df.columns else 0
    
    invalid_cpl = int((df['centipawn_loss'] < 0).sum()) if 'centipawn_loss' in df.columns else 0
    invalid_moves = missing_moves
    
    quality_summary = {
        "total_positions": total_positions,
        "unique_positions": unique_positions,
        "duplicate_positions": duplicate_positions,
        "unique_games": unique_games,
        "missing_fens": missing_fens,
        "invalid_fens": invalid_fens,
        "missing_moves": missing_moves,
        "missing_color": missing_color,
        "missing_engine_eval": missing_engine_eval,
        "missing_cpl": missing_cpl,
        "negative_cpl_errors": invalid_cpl,
        "invalid_moves": invalid_moves
    }
    
    out_path = os.path.join(output_dir, "summaries", "data_quality.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(quality_summary, f, indent=4)
        
    print(f"Data quality summary saved to {out_path}")
    
    if total_positions != 321339:
        print(f"WARNING: Expected 321,339 positions, but found {total_positions}!")
        
    return quality_summary
