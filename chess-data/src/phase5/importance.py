import pandas as pd
from sklearn.inspection import permutation_importance
import joblib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.phase4.models import prepare_features

def calculate_feature_importance(model_path, val_games_path, features_path, output_dir):
    print("Loading model for feature importance...")
    model = joblib.load(model_path)
    
    print("Loading validation dataset...")
    df = pd.read_parquet(features_path)
    
    with open(val_games_path) as f:
        text = f.read().strip()
        if '\\n' in text:
            val_games = set([x.strip() for x in text.split('\\n') if x.strip()])
        else:
            val_games = set([x.strip() for x in text.splitlines() if x.strip()])
        
    val_df = df[df['game_id'].astype(str).isin(val_games)]
    
    X_val, num_cols, cat_cols = prepare_features(val_df)
    features = num_cols + cat_cols
    X_val = X_val[features]
    y_val = val_df['label']
    
    if len(X_val) > 10000:
        sample_idx = X_val.sample(10000, random_state=42).index
        X_sample = X_val.loc[sample_idx]
        y_sample = y_val.loc[sample_idx]
    else:
        X_sample = X_val
        y_sample = y_val
        
    print("Calculating permutation importance (this may take a minute)...")
    r = permutation_importance(model, X_sample, y_sample, n_repeats=5, random_state=42, n_jobs=-1)
    
    features = num_cols + cat_cols
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': r.importances_mean,
        'Std': r.importances_std
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(f"{output_dir}/feature_importance.csv", index=False)
    
    top_10 = importance_df.head(10)
    top_10_dict = dict(zip(top_10['Feature'], top_10['Importance']))
    with open(f"{output_dir}/feature_importance.json", 'w') as f:
        json.dump(top_10_dict, f, indent=4)
        
    plt.figure(figsize=(10, 8))
    plt.barh(top_10['Feature'][::-1], top_10['Importance'][::-1], color='#2ECC71')
    plt.title('Top 10 Influential Features (Permutation Importance)')
    plt.xlabel('Mean Decrease in Accuracy')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_importance.png")
    plt.close()
    
    print("Feature importance calculated and saved.")
