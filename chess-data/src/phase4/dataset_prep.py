import pandas as pd
import numpy as np
import os

def prepare_candidate_dataset(df: pd.DataFrame, candidate_count=5):
    """
    Vectorized conversion of flat positional dataframe to candidate-ranking format.
    Ensures zero target leakage.
    """
    print(f"Preparing candidate ranking dataset (top {candidate_count} candidates)...")
    dfs = []
    
    # Ensure your_move_uci exists
    if 'your_move' in df.columns and 'your_move_uci' not in df.columns:
        df['your_move_uci'] = df['your_move']
        
    # Calculate baseline rank 1 eval
    df['r1_score'] = np.where(df['stockfish_rank_1_eval_cp'].notnull(), df['stockfish_rank_1_eval_cp'], 
                              np.where(df['stockfish_rank_1_eval_mate'] > 0, 10000, 
                              np.where(df['stockfish_rank_1_eval_mate'] < 0, -10000, 0)))
                              
    # Base cols to keep (avoid leakage)
    leak_cols = [
        'your_move_eval_cp', 'your_move_eval_mate', 'centipawn_loss', 'your_move_rank', 
        'is_engine_best', 'is_top3', 'is_top5', 'is_outside_top5', 
        'analysis_status', 'analysis_timestamp', 'eval_category', 'cluster', 'pca1', 'pca2',
        'r1_score' # dropped later
    ]
    
    base_cols = [c for c in df.columns if not c.startswith('stockfish_rank_') and not c.startswith('stockfish_best_') and c not in leak_cols]
    
    for rank in range(1, candidate_count + 1):
        move_col = f'stockfish_rank_{rank}_move'
        if move_col not in df.columns:
            move_col = f'stockfish_rank_{rank}_move_uci'
            
        cp_col = f'stockfish_rank_{rank}_eval_cp'
        mate_col = f'stockfish_rank_{rank}_eval_mate'
        
        if move_col not in df.columns: 
            print(f"Warning: {move_col} not found in columns.")
            continue
        
        subset = df[base_cols + [move_col, cp_col, mate_col, 'r1_score']].copy()
        subset = subset[subset[move_col].notnull() & (subset[move_col] != '')].copy()
        
        subset['candidate_move'] = subset[move_col]
        subset['stockfish_rank'] = rank
        
        # Calculate single unified candidate evaluation
        c_score = np.where(subset[cp_col].notnull(), subset[cp_col], 
                           np.where(subset[mate_col] > 0, 10000, 
                           np.where(subset[mate_col] < 0, -10000, 0)))
                           
        subset['candidate_eval'] = c_score
        
        # Drop relative to rank 1 (will be negative or zero)
        subset['eval_drop'] = c_score - subset['r1_score']
        
        subset['is_mate'] = subset[mate_col].notnull().astype(int)
        
        # LABEL
        subset['label'] = (subset['candidate_move'] == subset['your_move_uci']).astype(int)
        
        # Drop temp columns
        subset.drop(columns=[move_col, cp_col, mate_col, 'r1_score'], inplace=True)
        dfs.append(subset)
        
    cand_df = pd.concat(dfs, ignore_index=True)
    cand_df.sort_values(['position_id', 'stockfish_rank'], inplace=True)
    
    # Measure coverage
    pos_counts = cand_df.groupby('position_id')['label'].sum()
    covered = (pos_counts > 0).sum()
    total = len(pos_counts)
    coverage_pct = (covered / total) * 100 if total > 0 else 0
    print(f"Dataset coverage with Top {candidate_count}: {coverage_pct:.1f}% ({covered}/{total} positions)")
    
    return cand_df
