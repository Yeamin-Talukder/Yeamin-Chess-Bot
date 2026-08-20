import os
import time
import pandas as pd
import numpy as np

from src.phase5.importance import calculate_feature_importance
from src.phase5.simulator import run_style_sweep
from src.phase5.reporter import generate_reports
from src.bot.yeamin_bot import YeaminBot
import json

def main():
    print("="*60)
    print("PHASE 5: BUILD THE PLAYABLE 'YEAMIN CHESS BOT'")
    print("="*60)
    
    model_path = "data/phase4/models/yeamin_style_model.pkl"
    val_games_path = "data/phase3_5/features/validation_games.txt"
    test_games_path = "data/phase3_5/features/test_games.txt"
    features_path = "data/phase4/candidate_dataset_cache.parquet"
    out_dir = "data/phase5"
    
    os.makedirs(f"{out_dir}/feature_importance", exist_ok=True)
    os.makedirs(f"{out_dir}/experiments", exist_ok=True)
    os.makedirs(f"{out_dir}/charts", exist_ok=True)
    os.makedirs(f"{out_dir}/summaries", exist_ok=True)
    os.makedirs(f"{out_dir}/config", exist_ok=True)
    os.makedirs(f"{out_dir}/reports", exist_ok=True)
    
    engine_path = "bin/stockfish.exe"
    try:
        with open("config.json") as f:
            cfg = json.load(f)
            engine_path = cfg.get("stockfish_path", engine_path)
    except:
        pass
    
    print("\n1. Calculating Feature Importance...")
    calculate_feature_importance(model_path, val_games_path, features_path, f"{out_dir}/feature_importance")
    
    print("\n2. Running Style Sweep on Test Set...")
    best_config, res_df = run_style_sweep(model_path, features_path, test_games_path, out_dir)
    
    print("\n3. Running Latency Benchmark on 50 live positions...")
    bot = YeaminBot(engine_path=engine_path, model_path=model_path)
    
    df = pd.read_parquet(features_path, columns=['game_id', 'fen'])
    with open(test_games_path) as f:
        text = f.read().strip()
        if '\\n' in text:
            test_games = set([x.strip() for x in text.split('\\n') if x.strip()])
        else:
            test_games = set([x.strip() for x in text.splitlines() if x.strip()])
            
    test_df = df[df['game_id'].astype(str).isin(test_games)]
    sample_fens = test_df['fen'].unique()[:50]
    
    latencies = []
    for fen in sample_fens:
        start = time.perf_counter()
        bot.predict_move(fen, style_strength=best_config['style_strength'])
        end = time.perf_counter()
        latencies.append((end - start) * 1000)
        
    lat_stats = {
        'avg': np.mean(latencies),
        'p95': np.percentile(latencies, 95)
    }
    print(f"Average Latency: {lat_stats['avg']:.2f} ms")
    print(f"P95 Latency: {lat_stats['p95']:.2f} ms")
    
    model_size = os.path.getsize(model_path)
    try:
        engine_size = os.path.getsize(engine_path)
    except:
        engine_size = 0
        
    print("\n4. Generating Final Reports...")
    generate_reports(best_config, res_df, lat_stats, model_size, engine_size, out_dir)
    
    print("\n" + "="*60)
    print("PHASE 5 COMPLETE")
    print("="*60)
    print(f"Bot:\n    Yeamin Bot")
    print(f"\nBEST CONFIGURATION")
    print(f"Style Strength:\n    {best_config['style_strength']}")
    print(f"Average CPL Drop:\n    {best_config['avg_cpl_drop']:.2f} centipawns")
    print(f"Style Net:\n    {best_config['style_net']*100:.2f}%")
    print(f"Exact Imitation:\n    {best_config['exact_imitation']*100:.2f}%")
    print(f"\nDEPLOYMENT")
    print(f"Model Size:\n    {model_size / (1024*1024):.2f} MB")
    print(f"Average Inference Latency:\n    {lat_stats['avg']:.2f} ms")
    print("="*60)

if __name__ == "__main__":
    main()
