import pandas as pd
import chess
import multiprocessing
from tqdm import tqdm

def process_chunk(chunk):
    results = []
    for fen, uci in chunk:
        try:
            board = chess.Board(fen)
            move = chess.Move.from_uci(uci)
            
            is_capture = int(board.is_capture(move))
            is_castling = int(board.is_castling(move))
            is_promotion = int(move.promotion is not None)
            
            moving_piece = board.piece_type_at(move.from_square)
            is_pawn_move = int(moving_piece == chess.PAWN)
            
            # Simulate to see if it gives check
            board.push(move)
            is_check = int(board.is_check())
            
            results.append({
                'cand_is_capture': is_capture,
                'cand_is_castling': is_castling,
                'cand_is_promotion': is_promotion,
                'cand_is_check': is_check,
                'cand_is_pawn_move': is_pawn_move,
                'cand_moving_piece': moving_piece if moving_piece else 0
            })
        except Exception:
            results.append({
                'cand_is_capture': 0, 'cand_is_castling': 0, 'cand_is_promotion': 0, 
                'cand_is_check': 0, 'cand_is_pawn_move': 0, 'cand_moving_piece': 0
            })
    return results

def generate_candidate_features(df: pd.DataFrame, num_workers=None):
    """
    Given a candidate-exploded dataframe, extracts Python-chess features for each candidate.
    """
    print(f"Extracting candidate move properties for {len(df)} rows...")
    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() - 1)
        
    pairs = list(zip(df['fen'], df['candidate_move']))
    chunk_size = max(1, len(pairs) // (num_workers * 4))
    
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    
    results = []
    with multiprocessing.Pool(num_workers) as pool:
        for res in tqdm(pool.imap(process_chunk, chunks), total=len(chunks), desc="Parsing candidates"):
            results.extend(res)
            
    feats_df = pd.DataFrame(results)
    
    df = pd.concat([df.reset_index(drop=True), feats_df.reset_index(drop=True)], axis=1)
    return df
