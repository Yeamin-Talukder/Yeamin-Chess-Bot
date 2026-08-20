import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def analyze_stats(df: pd.DataFrame, output_dir: str):
    """
    Perform deep statistical analysis and clustering on style features.
    """
    print("Running statistical analysis...")
    out_dict = {}
    
    # Base masks
    w_mask = df['your_color'] == 'White'
    b_mask = df['your_color'] == 'Black'
    
    # 1. Performance Overview
    out_dict['performance'] = {
        'average_cpl': float(df['centipawn_loss'].mean()),
        'median_cpl': float(df['centipawn_loss'].median()),
        'top1_percentage': float(df['is_engine_best'].mean() * 100),
        'top3_percentage': float(df['is_top3'].mean() * 100),
        'top5_percentage': float(df['is_top5'].mean() * 100),
        'outside_top5_percentage': float(df['is_outside_top5'].mean() * 100)
    }
    
    # 2. Color Breakdown
    out_dict['color'] = {
        'white': {
            'average_cpl': float(df[w_mask]['centipawn_loss'].mean()),
            'top1_percentage': float(df[w_mask]['is_engine_best'].mean() * 100)
        },
        'black': {
            'average_cpl': float(df[b_mask]['centipawn_loss'].mean()),
            'top1_percentage': float(df[b_mask]['is_engine_best'].mean() * 100)
        }
    }
    
    # 3. Game Phase Breakdown
    phases = ['opening', 'middlegame', 'endgame']
    out_dict['game_phase'] = {}
    for p in phases:
        p_mask = df['game_phase'] == p
        if p_mask.sum() > 0:
            out_dict['game_phase'][p] = {
                'count': int(p_mask.sum()),
                'average_cpl': float(df[p_mask]['centipawn_loss'].mean()),
                'top1_percentage': float(df[p_mask]['is_engine_best'].mean() * 100)
            }
            
    # 4. Time Control Summary
    print("Analyzing Time Controls...")
    tc_summary = df.groupby('time_control').agg(
        count=('position_id', 'count'),
        avg_cpl=('centipawn_loss', 'mean'),
        median_cpl=('centipawn_loss', 'median'),
        top1_pct=('is_engine_best', lambda x: x.mean() * 100)
    ).reset_index().sort_values('count', ascending=False)
    tc_summary.to_csv(os.path.join(output_dir, "summaries", "time_control_summary.csv"), index=False)
    out_dict['time_control'] = tc_summary.head(5).to_dict(orient='records')
    
    # 5. Opening Summary
    print("Analyzing Openings...")
    op_summary = df.groupby('opening').agg(
        count=('position_id', 'count'),
        avg_cpl=('centipawn_loss', 'mean'),
        top1_pct=('is_engine_best', lambda x: x.mean() * 100)
    ).reset_index().sort_values('count', ascending=False)
    op_summary.to_csv(os.path.join(output_dir, "summaries", "opening_summary.csv"), index=False)
    
    # 6. Evaluate Position (Advantage / Disadvantage)
    print("Analyzing Decision-Making under pressure...")
    eval_conds = [
        (df['stockfish_best_eval_cp'] > 150),
        (df['stockfish_best_eval_cp'] >= 50) & (df['stockfish_best_eval_cp'] <= 150),
        (df['stockfish_best_eval_cp'] > -50) & (df['stockfish_best_eval_cp'] < 50),
        (df['stockfish_best_eval_cp'] >= -150) & (df['stockfish_best_eval_cp'] <= -50),
        (df['stockfish_best_eval_cp'] < -150)
    ]
    eval_choices = ['Winning', 'Advantage', 'Equal', 'Disadvantage', 'Losing']
    df['eval_category'] = np.select(eval_conds, eval_choices, default='Unknown')
    
    pressure_summary = df.groupby('eval_category').agg(
        count=('position_id', 'count'),
        avg_cpl=('centipawn_loss', 'mean')
    ).to_dict(orient='index')
    out_dict['pressure'] = pressure_summary

    # 7. Style Dimensions & PCA Clustering
    print("Clustering Playstyle...")
    try:
        # Group by Game ID to cluster games, not individual moves
        game_stats = df.groupby('game_id').agg(
            avg_cpl=('centipawn_loss', 'mean'),
            top1_rate=('is_engine_best', 'mean'),
            checks_pct=('is_check', 'mean'),
            avg_legal_moves=('legal_move_count', 'mean')
        ).fillna(0)
        
        if len(game_stats) > 10:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(game_stats)
            
            # PCA
            pca = PCA(n_components=2)
            pca_res = pca.fit_transform(scaled_features)
            game_stats['pca1'] = pca_res[:, 0]
            game_stats['pca2'] = pca_res[:, 1]
            
            # KMeans
            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            game_stats['cluster'] = kmeans.fit_predict(scaled_features)
            
            game_stats.to_csv(os.path.join(output_dir, "features", "game_clusters.csv"))
            out_dict['clusters_found'] = 4
    except Exception as e:
        print(f"Clustering failed (likely not enough numerical data or library error): {e}")
        out_dict['clusters_found'] = 0

    return df, out_dict
