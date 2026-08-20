import os
import sys
import pandas as pd
import json
import datetime

# Add local path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.phase35.validator import validate_data
from src.phase35.feature_eng import generate_features
from src.phase35.analyzer import analyze_stats
from src.phase35.visualizer import generate_charts
from src.phase35.reporter import generate_report

INPUT_FILE = "data/analysis/positions_stockfish.parquet"
OUTPUT_DIR = "data/phase3_5"

def main():
    print(f"Starting Phase 3.5 Personal Chess Style Analysis pipeline...")
    
    # Environment detection
    if 'google.colab' in sys.modules:
        print("Detected Google Colab environment. (Paths should be mounted correctly).")
        
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Cannot find master dataset at {INPUT_FILE}")
        sys.exit(1)
        
    print(f"Loading master dataset: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)
    
    # 1. Validate
    quality = validate_data(df, OUTPUT_DIR)
    
    # 2. Extract Features
    df = generate_features(df, OUTPUT_DIR)
    
    # 3. Analyze Stats & Clusters
    df, stats = analyze_stats(df, OUTPUT_DIR)
    
    # 4. Create Charts
    generate_charts(df, OUTPUT_DIR)
    
    # 5. Generate Report
    generate_report(stats, OUTPUT_DIR)
    
    # Save Run Metadata
    run_meta = {
        "timestamp": str(datetime.datetime.now()),
        "python_version": sys.version,
        "input_file": INPUT_FILE,
        "input_rows": len(df)
    }
    with open(os.path.join(OUTPUT_DIR, "summaries", "run_metadata.json"), "w") as f:
        json.dump(run_meta, f)
    
    # Final Output
    print("\\n==================================================")
    print("PHASE 3.5 COMPLETE")
    print("==================================================")
    print(f"Games analyzed: {quality['unique_games']}")
    print(f"Positions analyzed: {quality['total_positions']}")
    print(f"\\nAverage CPL: {stats['performance']['average_cpl']:.1f}")
    print(f"Median CPL: {stats['performance']['median_cpl']:.1f}")
    print(f"\\nStockfish agreement:")
    print(f"Top 1: {stats['performance']['top1_percentage']:.1f}%")
    print(f"Top 3: {stats['performance']['top3_percentage']:.1f}%")
    print(f"Top 5: {stats['performance']['top5_percentage']:.1f}%")
    
    print(f"\\nWhite:")
    print(f"Average CPL: {stats['color']['white']['average_cpl']:.1f}")
    print(f"Top 1: {stats['color']['white']['top1_percentage']:.1f}%")
    
    print(f"\\nBlack:")
    print(f"Average CPL: {stats['color']['black']['average_cpl']:.1f}")
    print(f"Top 1: {stats['color']['black']['top1_percentage']:.1f}%")
    
    # Find strongest/weakest phase
    phases = stats['game_phase']
    if phases:
        strongest = min(phases.keys(), key=lambda k: phases[k]['average_cpl'])
        weakest = max(phases.keys(), key=lambda k: phases[k]['average_cpl'])
        print(f"\\nStrongest phase:\\n{strongest.capitalize()}")
        print(f"Weakest phase:\\n{weakest.capitalize()}")
        
    print("\\nKey behavioral findings:")
    print("1. " + f"Your Top-3 engine agreement is {stats['performance']['top3_percentage']:.1f}%.")
    cpl_diff = stats['color']['white']['average_cpl'] - stats['color']['black']['average_cpl']
    pref = "White" if cpl_diff < 0 else "Black"
    print("2. " + f"You perform slightly better as {pref} (CPL difference of {abs(cpl_diff):.1f}).")
    if 'opening' in phases:
        print("3. " + f"Your opening accuracy is {phases['opening']['top1_percentage']:.1f}%.")
        
    print("\\nFiles created:")
    print("Style report:\\ndata/phase3_5/reports/personal_chess_style_report.html")
    print("Style profile:\\ndata/phase3_5/summaries/personal_style_profile.json")
    print("ML features:\\ndata/phase3_5/features/ml_training_features.parquet")
    print("Analysis features:\\ndata/phase3_5/features/style_features.parquet")
    print("==================================================")

if __name__ == "__main__":
    main()
