import pandas as pd
import time
import os
import sys
import numpy as np

from src.phase4.dataset_prep import prepare_candidate_dataset
from src.phase4.candidate_feats import generate_candidate_features
from src.phase4.models import get_splits, prepare_features, compute_time_weights, train_logistic_regression, train_hist_gradient_boosting
from src.phase4.evaluator import evaluate_model, log_experiments
from src.phase4.visualizer import generate_charts
from src.phase4.exporter import export_model
from src.phase4.reporter import generate_report

def main():
    print("="*60)
    print("PHASE 4: TRAIN THE 'WHAT WOULD I PLAY?' MODEL")
    print("="*60)
    
    data_path = "data/phase3_5/features/style_features.parquet"
    print(f"Loading {data_path}...")
    df_raw = pd.read_parquet(data_path)
    
    # 1. Dataset Preparation
    cand_df = prepare_candidate_dataset(df_raw, candidate_count=5)
    
    # 2. Candidate Features
    cand_df = generate_candidate_features(cand_df)
    
    # Save cache if needed
    cand_df.to_parquet("data/phase4/candidate_dataset_cache.parquet", index=False)
    
    # 3. Features & Splits
    cand_df, num_cols, cat_cols = prepare_features(cand_df)
    train_idx, val_idx, test_idx = get_splits(cand_df, "data/phase3_5/features")
    
    df_train = cand_df[train_idx].copy()
    df_test = cand_df[test_idx].copy()
    
    X_train = df_train[num_cols + cat_cols]
    y_train = df_train['label']
    X_test = df_test[num_cols + cat_cols]
    y_test = df_test['label']
    
    results = []
    
    # 4. Baselines
    print("Evaluating baselines on Test set...")
    res_rand = evaluate_model(df_test, 'Random')
    results.append(res_rand)
    
    res_sf = evaluate_model(df_test, 'Stockfish')
    results.append(res_sf)
    
    # 5. Logistic Regression
    lr_model = train_logistic_regression(X_train, y_train, num_cols, cat_cols)
    lr_probs = lr_model.predict_proba(X_test)[:, 1]
    res_lr = evaluate_model(df_test, 'Logistic Regression', lr_probs)
    results.append(res_lr)
    
    # 6. HistGBM with Time Decay Weighting
    t0 = time.time()
    
    # Calculate sample weights (more recent = higher weight)
    # df_train contains the 'date' column even though X_train does not
    base_weights = compute_time_weights(df_train)
    
    # Class balancing (positive instances are outnumbered 4:1)
    class_weights = np.where(y_train == 1, 4.0, 1.0)
    train_weights = base_weights * class_weights
    
    gbm_model = train_hist_gradient_boosting(
        X_train, y_train, num_cols, cat_cols, sample_weight=train_weights
    )
    t1 = time.time()
    
    # Batch predict latency
    t2 = time.time()
    gbm_probs = gbm_model.predict_proba(X_test)[:, 1]
    t3 = time.time()
    
    inf_lat = ((t3 - t2) / len(X_test)) * 1000 * 5 # average latency per position (5 candidates)
    
    res_gbm = evaluate_model(df_test, 'HistGradientBoosting', gbm_probs)
    results.append(res_gbm)
    
    # 7. Visualizations
    generate_charts(results, gbm_model.named_steps['clf'], X_train, y_train, df_test, num_cols, cat_cols, "data/phase4")
    
    # 8. Export & Report
    pkl_path = export_model(gbm_model, num_cols, cat_cols, results, "data/phase4")
    log_experiments(results, "data/phase4")
    generate_report(results, "data/phase4")
    
    model_size_mb = os.path.getsize(pkl_path) / (1024 * 1024)
    
    print("\n" + "="*60)
    print("PHASE 4 COMPLETE")
    print("="*60)
    
    print(f"Training positions: {train_idx.sum()//5}")
    print(f"Test positions: {test_idx.sum()//5}")
    print(f"Candidate coverage: Top 5: {res_gbm['candidate_coverage']:.1f}%")
    print(f"BEST MODEL: HistGradientBoosting")
    
    print("\nTest Performance:")
    print(f"Top-1: {res_gbm['top1']:.1f}%")
    print(f"Top-3: {res_gbm['top3']:.1f}%")
    print(f"Top-5: {res_gbm['top5']:.1f}%")
    print(f"MRR: {res_gbm['mrr']:.3f}")
    print(f"Log Loss: {res_gbm['log_loss']:.3f}")
    
    print(f"\nStockfish baseline Top-1: {res_sf['top1']:.1f}%")
    print(f"Personal deviation rate: {res_gbm['personal_deviation_rate']:.1f}%")
    print(f"Model correctly imitated my move when Stockfish disagreed: {res_gbm['cat_B']:.1f}%")
    
    print(f"Model size: {model_size_mb:.2f} MB")
    print(f"Average inference latency: {inf_lat:.2f} ms per position")
    
    print("\nModel saved: data/phase4/models/yeamin_style_model.pkl")
    print("Report saved: data/phase4/reports/phase4_model_report.html")
    print("="*60)

if __name__ == '__main__':
    main()
