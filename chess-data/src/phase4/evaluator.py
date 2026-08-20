import pandas as pd
import numpy as np
from sklearn.metrics import log_loss
import json
import os

def evaluate_model(df_test: pd.DataFrame, model_name: str, probabilities=None):
    """
    Evaluates a model or baseline on the test set, computing rigorous style metrics.
    """
    df = df_test.copy()
    if probabilities is not None:
        df['pred_prob'] = probabilities
    else:
        if model_name == 'Random':
            np.random.seed(42)
            df['pred_prob'] = np.random.rand(len(df))
        else:
            # Stockfish Baseline
            df['pred_prob'] = 1.0 / df['stockfish_rank']

    total_positions = df['position_id'].nunique()
    
    # Calculate Log Loss across all candidate rows
    try:
        loss = log_loss(df['label'], df['pred_prob'])
    except Exception:
        loss = float('nan')

    # Rank candidates by predicted probability per position
    df['pred_rank'] = df.groupby('position_id')['pred_prob'].rank(method='first', ascending=False)
    
    actual_moves = df[df['label'] == 1].copy()
    covered_positions = len(actual_moves)
    coverage = (covered_positions / total_positions) * 100 if total_positions > 0 else 0
    
    # Top-K accuracy (relative to ALL positions, not just covered)
    top1 = (actual_moves['pred_rank'] == 1).sum() / total_positions * 100
    top3 = (actual_moves['pred_rank'] <= 3).sum() / total_positions * 100
    top5 = (actual_moves['pred_rank'] <= 5).sum() / total_positions * 100
    mrr = (1.0 / actual_moves['pred_rank']).sum() / total_positions
    
    # Conditional Accuracy (only on covered positions)
    cond_top1 = (actual_moves['pred_rank'] == 1).mean() * 100 if covered_positions > 0 else 0
    
    # Style Imitation Test
    pred_top1 = df[df['pred_rank'] == 1].set_index('position_id')
    sf_top1 = df[df['stockfish_rank'] == 1].set_index('position_id')
    actual_df = df[df['label'] == 1].set_index('position_id')
    
    compare_df = pd.DataFrame(index=df['position_id'].unique())
    compare_df['actual'] = actual_df['candidate_move']
    compare_df['model_pred'] = pred_top1['candidate_move']
    compare_df['sf_pred'] = sf_top1['candidate_move']
    
    # Only compare where actual move is known in the candidate set
    compare_df_covered = compare_df.dropna(subset=['actual']).copy()
    
    if len(compare_df_covered) > 0:
        cat_A = ((compare_df_covered['model_pred'] == compare_df_covered['actual']) & (compare_df_covered['sf_pred'] == compare_df_covered['actual'])).mean() * 100
        cat_B = ((compare_df_covered['model_pred'] == compare_df_covered['actual']) & (compare_df_covered['sf_pred'] != compare_df_covered['actual'])).mean() * 100
        cat_C = ((compare_df_covered['model_pred'] != compare_df_covered['actual']) & (compare_df_covered['sf_pred'] == compare_df_covered['actual'])).mean() * 100
        cat_D = ((compare_df_covered['model_pred'] != compare_df_covered['actual']) & (compare_df_covered['sf_pred'] != compare_df_covered['actual'])).mean() * 100
    else:
        cat_A = cat_B = cat_C = cat_D = 0.0
        
    personal_dev = (compare_df['model_pred'] != compare_df['sf_pred']).mean() * 100

    return {
        'model': model_name,
        'top1': float(top1),
        'top3': float(top3),
        'top5': float(top5),
        'cond_top1': float(cond_top1),
        'mrr': float(mrr),
        'log_loss': float(loss),
        'candidate_coverage': float(coverage),
        'cat_A': float(cat_A),
        'cat_B': float(cat_B),
        'cat_C': float(cat_C),
        'cat_D': float(cat_D),
        'personal_deviation_rate': float(personal_dev)
    }

def log_experiments(results, output_dir):
    df = pd.DataFrame(results)
    out_path = os.path.join(output_dir, "experiments", "results.csv")
    df.to_csv(out_path, index=False)
    print(f"Experiment results saved to {out_path}")
