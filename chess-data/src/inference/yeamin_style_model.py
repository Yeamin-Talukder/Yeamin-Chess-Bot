import joblib
import os

class YeaminStyleModel:
    def __init__(self, model_path="data/phase4/models/yeamin_style_model.pkl"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Please run Phase 4.")
            
        self.model = joblib.load(model_path)
        
    def rank_candidates(self, features_df):
        """
        Takes a Pandas DataFrame prepared by `pipeline.py`.
        Returns the probability (0.0 to 1.0) for each candidate.
        """
        probs = self.model.predict_proba(features_df)[:, 1]
        return probs.tolist()
