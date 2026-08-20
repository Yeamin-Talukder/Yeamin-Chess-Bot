import pandas as pd
import numpy as np
import joblib
from src.phase4.models import prepare_features
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

def run_style_sweep(model_path, features_path, test_games_path, output_dir):
    print("Loading test dataset...")
    df = pd.read_parquet(features_path)
    
    with open(test_games_path) as f:
        text = f.read().strip()
        if '\\n' in text:
            test_games = set([x.strip() for x in text.split('\\n') if x.strip()])
        else:
            test_games = set([x.strip() for x in text.splitlines() if x.strip()])
        
    test_df = df[df['game_id'].astype(str).isin(test_games)].copy()
    
    X_test, num_cols, cat_cols = prepare_features(test_df.copy())
    
    print("Loading model and predicting base style probabilities...")
    model = joblib.load(model_path)
    probs = model.predict_proba(X_test)[:, 1]
    
    test_df['style_prob'] = probs
    
    def engine_win_prob(cp):
        cp = np.clip(cp, -10000, 10000)
        return 1.0 / (1.0 + 10.0 ** (-cp / 400.0))
        
    test_df['engine_prob'] = test_df['candidate_eval'].apply(engine_win_prob)
    
    strengths = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    max_cpl = 100
    
    results = []
    
    user_moves = test_df[test_df['label'] == 1]
    sf_matched_user = set(user_moves[user_moves['stockfish_rank'] == 1]['position_id'])
    
    print("Sweeping style strengths...")
    
    for s in strengths:
        print(f"Evaluating style strength: {s}")
        test_df['final_score'] = (1.0 - s) * test_df['engine_prob'] + s * test_df['style_prob']
        
        test_df.loc[test_df['eval_drop'] < -max_cpl, 'final_score'] = -1000.0
        
        idx_max = test_df.groupby('position_id')['final_score'].idxmax()
        picked_moves = test_df.loc[idx_max]
        
        exact_imitation = picked_moves['label'].mean()
        avg_cpl = picked_moves['eval_drop'].mean()
        median_cpl = picked_moves['eval_drop'].median()
        
        picked_matched_user = set(picked_moves[picked_moves['label'] == 1]['position_id'])
        
        style_wins = len(picked_matched_user - sf_matched_user)
        style_losses = len(sf_matched_user - picked_matched_user)
        total_positions = len(test_df['position_id'].unique())
        
        style_win_rate = style_wins / total_positions if total_positions else 0
        style_loss_rate = style_losses / total_positions if total_positions else 0
        style_net = style_win_rate - style_loss_rate
        
        sf_agreement = (picked_moves['stockfish_rank'] == 1).mean()
        blunder_rate = (picked_moves['eval_drop'] <= -300).mean()
        
        results.append({
            'style_strength': s,
            'exact_imitation': exact_imitation,
            'avg_cpl_drop': avg_cpl,
            'median_cpl_drop': median_cpl,
            'style_win_rate': style_win_rate,
            'style_loss_rate': style_loss_rate,
            'style_net': style_net,
            'stockfish_agreement': sf_agreement,
            'blunder_rate': blunder_rate
        })
        
    res_df = pd.DataFrame(results)
    res_df.to_csv(f"{output_dir}/experiments/style_strength_results.csv", index=False)
    
    best_config = res_df.loc[res_df['style_net'].idxmax()]
    if best_config['style_net'] <= 0:
        best_config = res_df.loc[res_df['style_strength'] == 0.5].iloc[0]
        
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['style_strength'], res_df['style_net'] * 100, marker='o', color='#2ECC71')
    plt.title('Style Net vs Style Strength')
    plt.xlabel('Style Strength')
    plt.ylabel('Style Net (%)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/charts/style_strength_vs_net.png")
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.plot(res_df['style_strength'], res_df['avg_cpl_drop'], marker='x', color='#C0392B')
    plt.title('Average CPL Drop vs Style Strength')
    plt.xlabel('Style Strength')
    plt.ylabel('CPL Drop (Centipawns)')
    plt.grid(True)
    plt.savefig(f"{output_dir}/charts/style_strength_vs_cpl.png")
    plt.close()
    
    return best_config.to_dict(), res_df
