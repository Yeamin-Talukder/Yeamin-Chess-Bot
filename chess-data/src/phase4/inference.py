import joblib
import pandas as pd
import chess
import numpy as np
import os
import sys

# Add root for cross-phase imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.phase35.feature_eng import analyze_fen

class YeaminStylePredictor:
    def __init__(self, model_dir):
        self.model = joblib.load(os.path.join(model_dir, "yeamin_style_model.pkl"))
        
    def predict_my_move(self, fen, stockfish_candidates, style_strength=1.0):
        """
        stockfish_candidates format: [{'move': 'e2e4', 'eval_cp': 50, 'eval_mate': None}, ...]
        Returns predicted move and ranking details.
        """
        if not stockfish_candidates:
            return {"predicted_move": None, "confidence": 0, "candidates": []}
            
        pos_features = analyze_fen(fen)
        
        rows = []
        r1_cp = stockfish_candidates[0].get('eval_cp')
        r1_mate = stockfish_candidates[0].get('eval_mate')
        r1_score = r1_cp if r1_cp is not None else (10000 if r1_mate and r1_mate > 0 else -10000 if r1_mate and r1_mate < 0 else 0)
        
        board = chess.Board(fen)
        
        for rank, cand in enumerate(stockfish_candidates, 1):
            move_uci = cand['move']
            c_cp = cand.get('eval_cp')
            c_mate = cand.get('eval_mate')
            
            c_score = c_cp if c_cp is not None else (10000 if c_mate and c_mate > 0 else -10000 if c_mate and c_mate < 0 else 0)
            
            try:
                move = chess.Move.from_uci(move_uci)
                is_capture = int(board.is_capture(move))
                is_castling = int(board.is_castling(move))
                is_promotion = int(move.promotion is not None)
                moving_piece = board.piece_type_at(move.from_square) or 0
                is_pawn_move = int(moving_piece == chess.PAWN)
                board.push(move)
                is_check = int(board.is_check())
                board.pop()
            except:
                is_capture = is_castling = is_promotion = moving_piece = is_pawn_move = is_check = 0
            
            row = pos_features.copy()
            row['candidate_move'] = move_uci
            row['stockfish_rank'] = rank
            row['candidate_eval'] = c_score
            row['eval_drop'] = c_score - r1_score
            row['is_mate'] = 1 if c_mate is not None else 0
            row['cand_is_capture'] = is_capture
            row['cand_is_castling'] = is_castling
            row['cand_is_promotion'] = is_promotion
            row['cand_is_check'] = is_check
            row['cand_is_pawn_move'] = is_pawn_move
            row['cand_moving_piece'] = moving_piece
            
            # Fill missing typical categorical cols with Unknown
            row['time_control'] = 'Unknown'
            row['opening'] = 'Unknown'
            row['eco'] = 'Unknown'
            row['your_color'] = 'White' if board.turn == chess.WHITE else 'Black'
            
            rows.append(row)
            
        df = pd.DataFrame(rows)
        for c in df.select_dtypes(exclude=[np.number]).columns:
            df[c] = df[c].fillna('Unknown').astype(str)
            
        # Get probabilities for class 1
        probs = self.model.predict_proba(df)[:, 1]
        
        # Style Strength Blending
        engine_probs = 1.0 / df['stockfish_rank'].values
        final_probs = (1 - style_strength) * engine_probs + style_strength * probs
        
        best_idx = np.argmax(final_probs)
        best_move = rows[best_idx]['candidate_move']
        
        result_cands = []
        for i, row in enumerate(rows):
            result_cands.append({
                "move": row['candidate_move'],
                "personal_probability": float(probs[i]),
                "blended_score": float(final_probs[i]),
                "stockfish_rank": row['stockfish_rank']
            })
            
        result_cands = sorted(result_cands, key=lambda x: x['blended_score'], reverse=True)
            
        return {
            "predicted_move": best_move,
            "confidence": float(final_probs[best_idx]),
            "candidates": result_cands
        }
