import pandas as pd
import numpy as np
import chess
import multiprocessing
from tqdm import tqdm
import os

PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0
}

def analyze_fen(fen):
    """
    Parses a single FEN string to extract material, game phase, and tactical features.
    """
    board = chess.Board(fen)
    
    # Material counts
    w_pawns = len(board.pieces(chess.PAWN, chess.WHITE))
    b_pawns = len(board.pieces(chess.PAWN, chess.BLACK))
    w_knights = len(board.pieces(chess.KNIGHT, chess.WHITE))
    b_knights = len(board.pieces(chess.KNIGHT, chess.BLACK))
    w_bishops = len(board.pieces(chess.BISHOP, chess.WHITE))
    b_bishops = len(board.pieces(chess.BISHOP, chess.BLACK))
    w_rooks = len(board.pieces(chess.ROOK, chess.WHITE))
    b_rooks = len(board.pieces(chess.ROOK, chess.BLACK))
    w_queens = len(board.pieces(chess.QUEEN, chess.WHITE))
    b_queens = len(board.pieces(chess.QUEEN, chess.BLACK))
    
    # Material score
    w_material = w_pawns + (w_knights * 3) + (w_bishops * 3) + (w_rooks * 5) + (w_queens * 9)
    b_material = b_pawns + (b_knights * 3) + (b_bishops * 3) + (b_rooks * 5) + (b_queens * 9)
    material_balance = w_material - b_material
    
    # Game Phase Logic
    # Opening: Both sides have castling rights, or total non-pawn material is very high.
    # Endgame: Both sides have no queens, or queens + < 1 minor piece. Usually < 14 points of non-pawn material.
    w_non_pawn = w_material - w_pawns
    b_non_pawn = b_material - b_pawns
    
    if w_non_pawn <= 13 and b_non_pawn <= 13:
        phase = 'endgame'
    elif (w_queens == 0 and b_queens == 0) and (w_non_pawn <= 15 and b_non_pawn <= 15):
        phase = 'endgame'
    elif board.fullmove_number <= 12:
        phase = 'opening'
    else:
        phase = 'middlegame'
        
    return {
        'w_pawns': w_pawns, 'b_pawns': b_pawns,
        'w_knights': w_knights, 'b_knights': b_knights,
        'w_bishops': w_bishops, 'b_bishops': b_bishops,
        'w_rooks': w_rooks, 'b_rooks': b_rooks,
        'w_queens': w_queens, 'b_queens': b_queens,
        'w_material': w_material, 'b_material': b_material,
        'material_balance': material_balance,
        'legal_move_count': board.legal_moves.count(),
        'is_check': board.is_check(),
        'can_castle_w': board.has_castling_rights(chess.WHITE),
        'can_castle_b': board.has_castling_rights(chess.BLACK),
        'game_phase': phase
    }

def process_fens_chunk(fens):
    return [analyze_fen(f) for f in fens]

def generate_features(df: pd.DataFrame, output_dir: str, num_workers=None):
    """
    Takes the master dataframe, extracts FEN features and ML target features,
    and returns an augmented dataframe.
    """
    print("Extracting FEN features... This might take a few minutes for 321k rows.")
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    fens = df['fen'].tolist()
    chunk_size = max(1, len(fens) // (num_workers * 4))
    
    chunks = [fens[i:i + chunk_size] for i in range(0, len(fens), chunk_size)]
    
    results = []
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap(process_fens_chunk, chunks), total=len(chunks), desc="Parsing FENs"):
            results.extend(res)
            
    fen_features_df = pd.DataFrame(results)
    
    # Merge back
    df = pd.concat([df.reset_index(drop=True), fen_features_df.reset_index(drop=True)], axis=1)
    
    print("Generating engine features...")
    # Clean rank
    df['your_move_rank'] = df['your_move_rank'].astype(str)
    df.loc[df['your_move_rank'] == '0', 'your_move_rank'] = '>5'
    df.loc[df['your_move_rank'] == '0.0', 'your_move_rank'] = '>5'
    
    df['is_engine_best'] = df['your_move_rank'].isin(['1', '1.0'])
    df['is_top3'] = df['your_move_rank'].isin(['1', '1.0', '2', '2.0', '3', '3.0'])
    df['is_top5'] = df['your_move_rank'].isin(['1', '1.0', '2', '2.0', '3', '3.0', '4', '4.0', '5', '5.0'])
    df['is_outside_top5'] = df['your_move_rank'] == '>5'
    
    # Save full style dataset
    style_path = os.path.join(output_dir, "features", "style_features.parquet")
    df.to_parquet(style_path, index=False)
    print(f"Full style features saved to {style_path}")
    
    # Create ML input dataset (avoiding leakage)
    print("Preparing ML Training Dataset (Pre-move features only)...")
    ml_cols_to_drop = [
        'your_move_eval_cp', 'your_move_eval_mate', 'centipawn_loss', 'your_move_rank',
        'is_engine_best', 'is_top3', 'is_top5', 'is_outside_top5',
        'stockfish_rank_1_move_uci', 'stockfish_rank_1_move_san', 'stockfish_best_move_uci', 'stockfish_best_move_san',
        'stockfish_rank_1_eval_cp', 'stockfish_rank_1_eval_mate', 'stockfish_best_eval_cp', 'stockfish_best_eval_mate',
        'stockfish_rank_2_move_uci', 'stockfish_rank_2_move_san', 'stockfish_rank_2_eval_cp', 'stockfish_rank_2_eval_mate',
        'stockfish_rank_3_move_uci', 'stockfish_rank_3_move_san', 'stockfish_rank_3_eval_cp', 'stockfish_rank_3_eval_mate',
        'stockfish_rank_4_move_uci', 'stockfish_rank_4_move_san', 'stockfish_rank_4_eval_cp', 'stockfish_rank_4_eval_mate',
        'stockfish_rank_5_move_uci', 'stockfish_rank_5_move_san', 'stockfish_rank_5_eval_cp', 'stockfish_rank_5_eval_mate',
        'stockfish_rank_1_move', 'stockfish_best_move', 'stockfish_rank_2_move', 'stockfish_rank_3_move', 'stockfish_rank_4_move', 'stockfish_rank_5_move',
        'analysis_status', 'analysis_timestamp'
    ]
    
    # Add target rename logic
    ml_df = df.drop(columns=[c for c in ml_cols_to_drop if c in df.columns]).copy()
    if 'your_move' in ml_df.columns and 'your_move_uci' not in ml_df.columns:
        ml_df = ml_df.rename(columns={'your_move': 'your_move_uci'})
        
    ml_path = os.path.join(output_dir, "features", "ml_training_features.parquet")
    ml_df.to_parquet(ml_path, index=False)
    print(f"ML pre-move features saved to {ml_path}")
    
    # Train / Val / Test Splits by GAME ID
    print("Generating Train/Val/Test Splits...")
    unique_games = df['game_id'].unique()
    np.random.seed(42)
    np.random.shuffle(unique_games)
    
    n_total = len(unique_games)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)
    
    train_games = unique_games[:n_train]
    val_games = unique_games[n_train:n_train + n_val]
    test_games = unique_games[n_train + n_val:]
    
    with open(os.path.join(output_dir, "features", "train_games.txt"), "w") as f:
        f.write("\\n".join(train_games))
    with open(os.path.join(output_dir, "features", "validation_games.txt"), "w") as f:
        f.write("\\n".join(val_games))
    with open(os.path.join(output_dir, "features", "test_games.txt"), "w") as f:
        f.write("\\n".join(test_games))
        
    print(f"Splits generated: {len(train_games)} Train | {len(val_games)} Val | {len(test_games)} Test")
    
    return df
