import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import os

def get_splits(df, phase3_5_features_dir):
    def read_games(filename):
        with open(os.path.join(phase3_5_features_dir, filename)) as f:
            text = f.read().strip()
            if '\\n' in text:
                return set([x.strip() for x in text.split('\\n') if x.strip()])
            else:
                return set([x.strip() for x in text.splitlines() if x.strip()])

    train_games = read_games("train_games.txt")
    val_games = read_games("validation_games.txt")
    test_games = read_games("test_games.txt")
        
    df_game_id_str = df['game_id'].astype(str)
    train_idx = df_game_id_str.isin(train_games)
    val_idx = df_game_id_str.isin(val_games)
    test_idx = df_game_id_str.isin(test_games)
    
    print(f"Games split: {len(train_games)} Train, {len(val_games)} Val, {len(test_games)} Test")
    print(f"Positions split: {train_idx.sum()//5} Train, {val_idx.sum()//5} Val, {test_idx.sum()//5} Test")
    
    return train_idx, val_idx, test_idx

def compute_time_weights(df, half_life_days=365):
    """
    Computes exponential decay weights based on the 'date' column.
    Recent games get weight ~1.0. Games 'half_life_days' old get weight ~0.5.
    """
    if 'date' not in df.columns:
        print("Warning: 'date' column not found, falling back to uniform weights.")
        return np.ones(len(df))
        
    # Convert to datetime and handle timezone/format parsing
    dates = pd.to_datetime(df['date'], errors='coerce', utc=True)
    
    # Fill missing dates with the median date
    if dates.isna().any():
        dates = dates.fillna(dates.median())
        
    # Find the most recent game date
    max_date = dates.max()
    
    # Calculate age in days
    age_days = (max_date - dates).dt.total_seconds() / (24 * 3600)
    
    # Exponential decay: weight = 0.5 ^ (age_days / half_life_days)
    weights = np.power(0.5, age_days / half_life_days)
    
    # Clip minimum weight to 0.1 so very old games still have a little influence
    weights = np.clip(weights, 0.1, 1.0)
    
    return weights.values

def prepare_features(df):
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
    
    num_cols = [c for c in num_cols if c in df.columns]
    cat_cols = [c for c in cat_cols if c in df.columns]
    
    for c in cat_cols:
        df[c] = df[c].fillna('Unknown').astype(str)
        
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0.0)
        
    return df, num_cols, cat_cols

def train_logistic_regression(X_train, y_train, num_cols, cat_cols):
    print("Training Logistic Regression...")
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ])
    
    model = Pipeline([
        ('preprocessor', preprocessor),
        ('clf', LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1))
    ])
    
    model.fit(X_train, y_train)
    return model

def train_hist_gradient_boosting(X_train, y_train, num_cols, cat_cols, sample_weight=None):
    print("Training HistGradientBoostingClassifier...")
    if sample_weight is not None:
        print(f"Using sample weights (min: {sample_weight.min():.3f}, max: {sample_weight.max():.3f}, mean: {sample_weight.mean():.3f})")
    
    # HistGBM only supports categorical cardinality <= 255.
    # eco and opening have higher cardinality, so we treat them as continuous (ordinal encoded).
    cat_features_indices = []
    for i, col in enumerate(cat_cols):
        if col not in ['eco', 'opening']:
            cat_features_indices.append(len(num_cols) + i)
            
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), cat_cols)
        ])
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('clf', HistGradientBoostingClassifier(
            categorical_features=cat_features_indices,
            max_iter=800,
            learning_rate=0.05,
            max_leaf_nodes=127,
            l2_regularization=0.1,
            random_state=42
        ))
    ])
    
    if sample_weight is not None:
        model.fit(X_train, y_train, clf__sample_weight=sample_weight)
    else:
        model.fit(X_train, y_train)
    return model
