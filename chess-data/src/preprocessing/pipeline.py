import pandas as pd

def build_inference_dataframe(position_features, candidates_features_list):
    """
    Combines position dict and list of candidate dicts into a properly structured DataFrame.
    """
    rows = []
    
    for cand in candidates_features_list:
        row = position_features.copy()
        row.update(cand)
        
        # Add required categorical defaults if missing
        if 'time_control' not in row:
            row['time_control'] = 'Unknown'
        if 'eco' not in row:
            row['eco'] = 'Unknown'
        if 'opening' not in row:
            row['opening'] = 'Unknown'
            
        rows.append(row)
        
    df = pd.DataFrame(rows)
    
    # Apply standard whitelists exactly as in Phase 4 models.py
    num_cols = [
        'move_number', 'w_pawns', 'b_pawns', 'w_knights', 'b_knights',
        'w_bishops', 'b_bishops', 'w_rooks', 'b_rooks', 'w_queens', 'b_queens',
        'w_material', 'b_material', 'material_balance', 'legal_move_count', 
        'stockfish_rank', 'candidate_eval', 'eval_drop', 'is_mate',
        'cand_is_capture', 'cand_is_castling', 'cand_is_promotion', 
        'cand_is_check', 'cand_is_pawn_move'
    ]
    
    cat_cols = [
        'side_to_move', 'time_control', 'eco', 'opening', 'game_phase', 'cand_moving_piece'
    ]
    
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0.0
            
    for c in cat_cols:
        if c not in df.columns:
            df[c] = 'Unknown'
            
    df = df[num_cols + cat_cols].copy()
    
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str)
        
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
    return df
